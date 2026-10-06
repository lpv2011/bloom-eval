"""`bloom-eval doctor`: compare our configured endpoints with Bloom's live OpenAPI spec."""
from __future__ import annotations

import json
import re
from pathlib import Path

import httpx

from .client import BloomClient
from .config import Endpoints

_PARAM = re.compile(r"\{[^}]+\}")


def _norm(path: str) -> str:
    return _PARAM.sub("{}", path.rstrip("/"))


def _resolve(spec: dict, schema: dict) -> dict:
    if "$ref" in schema:
        node = spec
        for part in schema["$ref"].lstrip("#/").split("/"):
            node = node[part]
        return node
    return schema


def run_doctor(ep: Endpoints, spec_path: str | None = None) -> int:
    if spec_path:
        spec = json.loads(Path(spec_path).read_text())
    else:
        url = ep.base_url + "/spec.json"
        print(f"fetching {url}")
        r = httpx.get(url, timeout=30, follow_redirects=True)
        r.raise_for_status()
        spec = r.json()

    ops = {}
    for path, item in spec.get("paths", {}).items():
        for method, op in item.items():
            if method.lower() in ("get", "post", "put", "patch", "delete"):
                ops[(method.upper(), _norm(path))] = (path, op)

    want = [
        ("GET", ep.image_path), ("GET", ep.credits_path), ("POST", ep.reviews_path),
        ("GET", ep.reviews_path + "/{id}"), ("GET", ep.brand_path),
        ("POST", ep.generate_path), ("GET", ep.skill_path),
    ]
    bad = 0
    print("\nconfigured endpoints vs spec:")
    for method, path in want:
        ok = (method, _norm(path)) in ops
        bad += not ok
        print(f"  {'OK ' if ok else 'MISSING'}  {method:5s} {path}")

    print("\nspec operations that look relevant (images / brands / skill):")
    for (method, _), (path, op) in sorted(ops.items()):
        if re.search(r"image|skill|brand", path) and "wait" not in path:
            print(f"  {method:5s} {path:55s} {op.get('summary', '')[:60]}")

    gen = next((op for (m, p), (path, op) in ops.items() if m == "POST" and re.search(r"image", path) and re.search(r"generat", path + op.get("summary", ""), re.I) and "edit" not in path), None)
    if gen:
        body = gen.get("requestBody", {}).get("content", {}).get("application/json", {}).get("schema", {})
        props = _resolve(spec, body).get("properties", {})
        print("\nrequest fields of the generate operation:", ", ".join(props) or "(none found)")
        print("compare these with f_prompt / f_aspect / f_size / f_variants / f_refs in your endpoints file")

    if bad:
        print(f"\n{bad} configured endpoint(s) not in spec: write an endpoints.yaml override and pass --endpoints.")
    else:
        print("\nall configured endpoints exist in the spec.")

    try:
        c = BloomClient(ep)
        print("\ncredits:", json.dumps(c.credits())[:200])
    except SystemExit:
        print("\n(BLOOM_API_KEY not set; skipped credits check)")
    except Exception as exc:
        print(f"\ncredits check failed: {exc}")
    return 1 if bad else 0
