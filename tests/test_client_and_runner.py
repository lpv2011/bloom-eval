"""Offline end-to-end: a mock Bloom API (httpx.MockTransport) feeds the real client + runner."""
import io
import json

import httpx
import yaml
from PIL import Image

from bloom_eval.client import BloomClient, extract_ids
from bloom_eval.config import Endpoints
from bloom_eval.report import aggregate, load_run
from bloom_eval.runner import find_key, find_palette, load_suite, run_suite
from bloom_eval.synth import make_banner, make_logo

PAL = ["#0B3D2E", "#F4E9D8"]


def test_extract_ids_is_tolerant():
    assert extract_ids({"data": {"imageIds": ["a", "b"]}}) == ["a", "b"]
    assert extract_ids({"data": {"images": [{"id": "x"}]}}) == ["x"]
    assert extract_ids({"data": {"id": "solo"}}) == ["solo"]


def test_find_palette_and_skill_id_without_assuming_layout():
    skill = {"profile": {"palette": {"primary": "#0B3D2E", "list": ["#F4E9D8", "#123"]}}, "skill": {"skillId": "sk_1"}}
    assert find_palette(skill) == ["#0B3D2E", "#F4E9D8", "#123"]
    assert find_key(skill, "skillId") == "sk_1"


def _mock_transport(png_bytes: bytes):
    state = {"n": 0}

    def handler(req: httpx.Request) -> httpx.Response:
        assert req.headers["x-api-key"] == "test"
        path = req.url.path
        if req.method == "POST" and path.endswith("/images/generate"):
            body = json.loads(req.content)
            assert body["brandSessionId"] == "brand-1"
            state["n"] += 1
            return httpx.Response(202, json={"data": {"imageIds": [f"img-{state['n']}"]}})
        if path.startswith("/api/v1/images/"):
            iid = path.rsplit("/", 1)[-1]
            return httpx.Response(200, json={"data": {"id": iid, "status": "completed", "imageUrl": f"https://cdn.test/{iid}.png"}})
        if path.endswith("/skill"):
            return httpx.Response(200, json={"data": {"skillId": "sk_demo", "profile": {"palette": PAL}}})
        if path.endswith("/credits"):
            return httpx.Response(200, json={"data": {"balance": 100}})
        return httpx.Response(404)

    return httpx.MockTransport(handler)


def test_full_run_offline(tmp_path, monkeypatch):
    logo = make_logo(*PAL)
    buf = io.BytesIO()
    make_banner(PAL, logo, "Meet your morning", seed=3).save(buf, "PNG")
    png = buf.getvalue()
    logo.save(tmp_path / "logo.png")

    suite_path = tmp_path / "suite.yaml"
    suite_path.write_text(yaml.safe_dump({
        "brand": {"id": "brand-1", "logo": "logo.png"},
        "samples": 2, "checks": ["color", "logo"],
        "tasks": [{"id": "hero", "prompt": "p", "aspect_ratio": "16:9"}],
    }))

    client = BloomClient(Endpoints(), api_key="test", transport=_mock_transport(png))
    monkeypatch.setattr(client, "download", lambda url: png)

    out = run_suite(client, load_suite(suite_path), tmp_path / "runs", workers=2)
    run = load_run(out)
    assert run["skill"]["skill_id"] == "sk_demo"
    assert len(run["results"]) == 2 and all("checks" in r for r in run["results"])
    agg = aggregate(run)
    assert agg["color"]["pass"] == 2 and agg["logo"]["pass"] == 2
