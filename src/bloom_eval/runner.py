"""Run a suite: snapshot the Skill, generate N samples per task, score each image."""
from __future__ import annotations

import hashlib
import io
import json
import re
import time
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import yaml
from PIL import Image

from .checks import check_color, check_logo, check_text
from .models import CheckResult

_HEX = re.compile(r"^#?(?:[0-9a-fA-F]{3}|[0-9a-fA-F]{6})$")


def load_suite(path: str | Path) -> dict:
    suite = yaml.safe_load(Path(path).read_text())
    suite["_dir"] = str(Path(path).resolve().parent)
    suite.setdefault("samples", 3)
    suite.setdefault("checks", ["color", "text", "logo"])
    if not suite.get("tasks"):
        raise ValueError("suite has no tasks")
    return suite


# ---- tolerant helpers: we do not assume the Skill JSON layout ------------------
def find_key(obj: Any, key: str) -> Any:
    if isinstance(obj, dict):
        if key in obj and isinstance(obj[key], (str, int)):
            return obj[key]
        for v in obj.values():
            r = find_key(v, key)
            if r is not None:
                return r
    elif isinstance(obj, list):
        for v in obj:
            r = find_key(v, key)
            if r is not None:
                return r
    return None


def _hexes(obj: Any) -> list[str]:
    if isinstance(obj, str):
        return [obj if obj.startswith("#") else "#" + obj] if _HEX.match(obj) else []
    out: list[str] = []
    if isinstance(obj, dict):
        for v in obj.values():
            out += _hexes(v)
    elif isinstance(obj, list):
        for v in obj:
            out += _hexes(v)
    return out


def find_palette(profile: Any) -> list[str]:
    """Collect hex colours found under any key whose name contains 'palette' or 'colo(u)r'."""
    found: list[str] = []

    def walk(o: Any) -> None:
        if isinstance(o, dict):
            for k, v in o.items():
                if any(t in k.lower() for t in ("palette", "color", "colour")):
                    found.extend(_hexes(v))
                else:
                    walk(v)
        elif isinstance(o, list):
            for v in o:
                walk(v)

    walk(profile)
    return list(dict.fromkeys(found))


# ---- scoring -------------------------------------------------------------------
def build_context(suite: dict, skill: dict | None) -> dict:
    brand = suite.get("brand", {})
    palette = brand.get("palette") or (find_palette(skill) if skill else [])
    logo = None
    if brand.get("logo"):
        logo = Image.open(Path(suite["_dir"]) / brand["logo"])
        logo.load()
    return {"palette": palette, "logo": logo}


def score_image(img: Image.Image, task: dict, suite: dict, ctx: dict) -> list[CheckResult]:
    out: list[CheckResult] = []
    for name in suite["checks"]:
        if name == "color":
            out.append(check_color(img, ctx["palette"], **suite.get("color", {})))
        elif name == "text":
            out.append(check_text(img, task.get("expected_text"), **suite.get("text", {})))
        elif name == "logo":
            out.append(check_logo(img, ctx["logo"], **suite.get("logo_opts", {})))
        else:
            raise ValueError(f"unknown check {name!r}")
    return out


def _parse_review(data: dict) -> list[CheckResult]:
    """Map Bloom Review verdicts to CheckResults. Tolerant: unknown shapes are kept raw only."""
    res = data.get("results", data.get("checks"))
    items: list[tuple[str, dict]] = []
    if isinstance(res, dict):
        items = [(k, v) for k, v in res.items() if isinstance(v, dict)]
    elif isinstance(res, list):
        items = [(str(v.get("check") or v.get("name")), v) for v in res if isinstance(v, dict)]
    out = []
    for name, v in items:
        verdict = v.get("verdict")
        out.append(
            CheckResult(
                f"review:{name}",
                True if verdict == "pass" else False if verdict == "fail" else None,
                1.0 if verdict == "pass" else 0.0,
                {"verdict": verdict, "note": v.get("note") or v.get("reason")},
            )
        )
    return out


# ---- run -----------------------------------------------------------------------
def run_suite(client, suite: dict, out_root: str | Path = "runs", *, with_review: bool = False, workers: int = 4) -> Path:
    brand_id = suite["brand"]["id"]
    run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    out = Path(out_root) / run_id
    (out / "images").mkdir(parents=True)

    skill = None
    try:
        skill = client.get_skill(brand_id)
    except Exception as exc:  # path unverified or no access: keep going, palette can come from the suite
        print(f"[warn] could not read Brand Skill ({exc}); using suite.brand.palette only")
    snap = json.dumps(skill, sort_keys=True) if skill is not None else ""
    skill_info = {
        "skill_id": find_key(skill, "skillId") if skill else None,
        "snapshot_sha256": hashlib.sha256(snap.encode()).hexdigest() if skill else None,
    }
    if skill is not None:
        (out / "skill_snapshot.json").write_text(json.dumps(skill, indent=2))

    ctx = build_context(suite, skill)
    if "color" in suite["checks"] and not ctx["palette"]:
        print("[warn] no palette found; colour check will be 'uncertain'. Set brand.palette in the suite.")

    jobs = [(t, i) for t in suite["tasks"] for i in range(suite["samples"])]

    def one(job: tuple[dict, int]) -> dict:
        task, i = job
        rec: dict[str, Any] = {"task_id": task["id"], "sample": i}
        try:
            ids = client.generate_image(
                brand_id,
                task["prompt"],
                aspect_ratio=task.get("aspect_ratio"),
                size=task.get("size", suite.get("size")),
                reference_image_ids=task.get("reference_image_ids"),
            )
            rec["image_id"] = ids[0]
            data = client.wait_image(ids[0])
            if data.get("status") != "completed":
                rec["generation_failed"] = data.get("failureReason") or data.get("status")
                return rec
            blob = client.download(data["imageUrl"])
            path = out / "images" / f"{task['id']}_{i}.png"
            img = Image.open(io.BytesIO(blob)).convert("RGB")
            img.save(path)
            rec["image_path"] = str(path.relative_to(out))
            checks = score_image(img, task, suite, ctx)
            if with_review:
                rid = client.create_review(brand_id, ids[0], ["color", "typography", "logo", "copy"], task.get("expected_text"))
                rv = client.wait_review(rid)
                rec["review_raw"] = rv
                checks += _parse_review(rv)
            rec["checks"] = [c.to_dict() for c in checks]
        except Exception as exc:  # one bad sample must not kill the run
            rec["error"] = f"{type(exc).__name__}: {exc}"[:300]
        return rec

    with ThreadPoolExecutor(max_workers=workers) as ex:
        results = list(ex.map(one, jobs))

    (out / "results.json").write_text(
        json.dumps(
            {
                "run_id": run_id,
                "created": time.time(),
                "brand_id": brand_id,
                "skill": skill_info,
                "suite": {k: v for k, v in suite.items() if not k.startswith("_")},
                "results": results,
            },
            indent=2,
        )
    )
    return out


def score_folder(folder: str | Path, suite: dict, out_root: str | Path = "runs") -> Path:
    """Score existing images (e.g. exported from Bloom by hand). File stem must be '<task_id>_<n>'."""
    ctx = build_context(suite, None)
    tasks = {t["id"]: t for t in suite["tasks"]}
    run_id = "local-" + datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    out = Path(out_root) / run_id
    (out / "images").mkdir(parents=True)
    results = []
    for p in sorted(Path(folder).glob("*.*")):
        tid, _, n = p.stem.rpartition("_")
        if tid not in tasks:
            continue
        img = Image.open(p).convert("RGB")
        img.save(out / "images" / f"{p.stem}.png")
        checks = score_image(img, tasks[tid], suite, ctx)
        results.append(
            {"task_id": tid, "sample": int(n) if n.isdigit() else 0, "image_path": f"images/{p.stem}.png",
             "checks": [c.to_dict() for c in checks]}
        )
    (out / "results.json").write_text(
        json.dumps({"run_id": run_id, "brand_id": suite.get("brand", {}).get("id"), "skill": {}, "results": results}, indent=2)
    )
    return out
