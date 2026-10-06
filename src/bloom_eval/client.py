"""Minimal Bloom REST client: x-api-key auth, shared rate limiter, retries.

Documented limits: 120 accepted requests / 60 s per API key. We stay under that with a
sliding-window limiter and honour Retry-After on 429. 402 (credits) is surfaced, never
retried: the docs say to branch on `code` (INSUFFICIENT_CREDITS vs PAYMENT_REQUIRED).
"""
from __future__ import annotations

import os
import threading
import time
from collections import deque
from typing import Any

import httpx

from .config import Endpoints


class BloomError(RuntimeError):
    def __init__(self, status: int, code: str | None, message: str):
        super().__init__(f"HTTP {status} {code or ''} {message}".strip())
        self.status, self.code = status, code


def _error_code(r: httpx.Response) -> str | None:
    """Best-effort extraction of a machine-readable error code (e.g. INSUFFICIENT_CREDITS)."""
    try:
        body = r.json()
    except ValueError:
        return None
    if not isinstance(body, dict):
        return None
    err = body.get("error")
    if isinstance(err, dict) and err.get("code"):
        return str(err["code"])
    return str(body["code"]) if body.get("code") else None


class RateLimiter:
    def __init__(self, max_calls: int = 100, period: float = 60.0):
        self.max_calls, self.period = max_calls, period
        self._t: deque[float] = deque()
        self._lock = threading.Lock()

    def acquire(self) -> None:
        while True:
            with self._lock:
                now = time.monotonic()
                while self._t and now - self._t[0] >= self.period:
                    self._t.popleft()
                if len(self._t) < self.max_calls:
                    self._t.append(now)
                    return
                wait = self.period - (now - self._t[0])
            time.sleep(max(wait, 0.05))


def extract_ids(payload: Any) -> list[str]:
    """Pull image IDs out of a generate response without assuming one exact shape."""
    data = payload.get("data", payload) if isinstance(payload, dict) else payload
    if isinstance(data, dict):
        for key in ("imageIds", "ids", "image_ids"):
            if isinstance(data.get(key), list):
                return [str(x) for x in data[key]]
        for key in ("images", "items", "results"):
            if isinstance(data.get(key), list):
                return [str(x["id"] if isinstance(x, dict) else x) for x in data[key]]
        if "id" in data:
            return [str(data["id"])]
    if isinstance(data, list):
        return [str(x["id"] if isinstance(x, dict) else x) for x in data]
    raise ValueError(f"could not find image ids in response: {str(payload)[:300]}")


class BloomClient:
    def __init__(
        self,
        endpoints: Endpoints | None = None,
        api_key: str | None = None,
        timeout: float = 320.0,
        transport: httpx.BaseTransport | None = None,
    ):
        self.ep = endpoints or Endpoints()
        key = api_key or os.environ.get("BLOOM_API_KEY")
        if not key:
            raise SystemExit("Set BLOOM_API_KEY (Bloom account -> API keys).")
        self._http = httpx.Client(
            base_url=self.ep.base_url,
            headers={"x-api-key": key},
            timeout=timeout,
            follow_redirects=True,
            transport=transport,
        )
        self._limiter = RateLimiter()

    def _request(self, method: str, path: str, *, retries: int = 4, **kw) -> dict:
        for attempt in range(retries + 1):
            self._limiter.acquire()
            try:
                r = self._http.request(method, path, **kw)
            except httpx.TransportError:
                if attempt == retries:
                    raise
                time.sleep(2**attempt)
                continue
            if r.status_code == 429 and attempt < retries:
                time.sleep(float(r.headers.get("Retry-After", 60)))
                continue
            if r.status_code >= 500 and attempt < retries:
                time.sleep(2**attempt)
                continue
            if r.status_code >= 400:
                raise BloomError(r.status_code, _error_code(r), r.text[:300])
            return r.json() if r.content else {}
        raise RuntimeError("unreachable")

    # ---- account -------------------------------------------------------------
    def credits(self, workspace_id: str | None = None) -> dict:
        params = {"workspaceId": workspace_id} if workspace_id else None
        return self._request("GET", self.ep.credits_path, params=params)

    # ---- brand ---------------------------------------------------------------
    def get_brand(self, brand_id: str) -> dict:
        return self._request("GET", self.ep.brand_path.format(id=brand_id))

    def get_skill(self, brand_id: str) -> dict:
        return self._request("GET", self.ep.skill_path.format(id=brand_id))

    # ---- images --------------------------------------------------------------
    def generate_image(
        self,
        brand_id: str,
        prompt: str,
        *,
        aspect_ratio: str | None = None,
        size: str | None = None,
        variants: int | None = None,
        reference_image_ids: list[str] | None = None,
        extra: dict | None = None,
    ) -> list[str]:
        ep = self.ep
        body: dict[str, Any] = {ep.f_brand: brand_id, ep.f_prompt: prompt}
        if aspect_ratio:
            body[ep.f_aspect] = aspect_ratio
        if size:
            body[ep.f_size] = size
        if variants:
            body[ep.f_variants] = variants
        if reference_image_ids:
            body[ep.f_refs] = reference_image_ids
        body.update(extra or {})
        return extract_ids(self._request("POST", ep.generate_path, json=body))

    def wait_image(self, image_id: str, max_wait: float = 600.0) -> dict:
        """Poll GET /images/{id}?wait=true until completed/failed."""
        deadline = time.monotonic() + max_wait
        while True:
            res = self._request("GET", self.ep.image_path.format(id=image_id), params={"wait": "true"})
            data = res.get("data", res)
            if data.get("status") in ("completed", "failed"):
                return data
            if time.monotonic() > deadline:
                raise TimeoutError(f"image {image_id} still {data.get('status')} after {max_wait}s")

    def download(self, image_url: str) -> bytes:
        url = image_url if image_url.startswith("http") else self.ep.site_root + image_url
        r = httpx.get(url, follow_redirects=True, timeout=120)
        r.raise_for_status()
        return r.content

    # ---- reviews (private beta) ----------------------------------------------
    def create_review(self, brand_id: str, image_id: str, checks: list[str], notes: str | None = None) -> str:
        body: dict[str, Any] = {
            "brandSessionId": brand_id,
            "assets": [{"kind": "image", "imageId": image_id}],
            "checks": checks,
        }
        if notes:
            body["context"] = {"notes": notes}
        res = self._request("POST", self.ep.reviews_path, json=body)
        return (res.get("data", res))["id"]

    def wait_review(self, review_id: str, max_wait: float = 600.0) -> dict:
        deadline = time.monotonic() + max_wait
        while True:
            res = self._request("GET", f"{self.ep.reviews_path}/{review_id}", params={"wait": "true"})
            data = res.get("data", res)
            if data.get("status") not in ("pending", "processing"):
                return data
            if time.monotonic() > deadline:
                raise TimeoutError("review timed out")
