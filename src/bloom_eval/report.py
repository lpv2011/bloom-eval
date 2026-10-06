"""Aggregate, compare and render runs. Pass rates exclude 'uncertain'; those are reported separately."""
from __future__ import annotations

import html
import json
import math
from pathlib import Path


def load_run(path: str | Path) -> dict:
    p = Path(path)
    p = p / "results.json" if p.is_dir() else p
    run = json.loads(p.read_text())
    run["_dir"] = str(p.parent)
    return run


def aggregate(run: dict) -> dict[str, dict[str, int]]:
    agg: dict[str, dict[str, int]] = {}
    for r in run["results"]:
        for c in r.get("checks", []):
            a = agg.setdefault(c["name"], {"pass": 0, "fail": 0, "uncertain": 0})
            a["uncertain" if c["passed"] is None else "pass" if c["passed"] else "fail"] += 1
    return agg


def gen_failures(run: dict) -> int:
    return sum(1 for r in run["results"] if r.get("generation_failed") or r.get("error"))


def wilson(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    if n == 0:
        return (0.0, 1.0)
    p = k / n
    d = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / d
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return (max(0.0, centre - half), min(1.0, centre + half))


def two_prop_p(k1: int, n1: int, k2: int, n2: int) -> float:
    """Two-sided two-proportion z-test p-value (pooled)."""
    if n1 == 0 or n2 == 0:
        return 1.0
    p = (k1 + k2) / (n1 + n2)
    se = math.sqrt(p * (1 - p) * (1 / n1 + 1 / n2))
    if se == 0:
        return 1.0
    z = (k2 / n2 - k1 / n1) / se
    return math.erfc(abs(z) / math.sqrt(2))


def compare(a: dict, b: dict, *, alpha: float = 0.05, min_drop: float = 0.05) -> list[dict]:
    """Rows comparing run B (candidate) against run A (baseline). regression=True if B is
    significantly worse by at least `min_drop` (absolute pass-rate points)."""
    ga, gb = aggregate(a), aggregate(b)
    rows = []
    for name in sorted(set(ga) | set(gb)):
        sa = ga.get(name, {"pass": 0, "fail": 0, "uncertain": 0})
        sb = gb.get(name, {"pass": 0, "fail": 0, "uncertain": 0})
        na, nb = sa["pass"] + sa["fail"], sb["pass"] + sb["fail"]
        ra = sa["pass"] / na if na else float("nan")
        rb = sb["pass"] / nb if nb else float("nan")
        p = two_prop_p(sa["pass"], na, sb["pass"], nb)
        delta = rb - ra if na and nb else float("nan")
        rows.append(
            {
                "check": name, "n_a": na, "n_b": nb, "rate_a": ra, "rate_b": rb, "delta": delta, "p": p,
                "ci_a": wilson(sa["pass"], na), "ci_b": wilson(sb["pass"], nb),
                "unc_a": sa["uncertain"], "unc_b": sb["uncertain"],
                "regression": bool(na and nb and delta <= -min_drop and p < alpha),
            }
        )
    return rows


def format_compare(rows: list[dict], a: dict, b: dict) -> str:
    def pct(x):
        return "  n/a " if x != x else f"{x * 100:5.1f}%"

    sa, sb = a.get("skill", {}).get("skill_id"), b.get("skill", {}).get("skill_id")
    lines = [
        f"baseline  {a['run_id']}  skill={sa}",
        f"candidate {b['run_id']}  skill={sb}",
        "",
        f"{'check':18s} {'base':>7s} {'cand':>7s} {'delta':>8s} {'p':>7s}  verdict",
    ]
    for r in rows:
        v = "REGRESSION" if r["regression"] else ("improved" if r["delta"] == r["delta"] and r["delta"] > 0 and r["p"] < 0.05 else "ok")
        d = "  n/a  " if r["delta"] != r["delta"] else f"{r['delta'] * 100:+6.1f}pp"
        lines.append(f"{r['check']:18s} {pct(r['rate_a'])} {pct(r['rate_b'])} {d:>8s} {r['p']:7.3f}  {v}")
    return "\n".join(lines)


def render_html(run: dict) -> str:
    agg = aggregate(run)
    rows = ""
    for name, s in sorted(agg.items()):
        n = s["pass"] + s["fail"]
        lo, hi = wilson(s["pass"], n)
        rate = f"{s['pass'] / n * 100:.0f}%" if n else "n/a"
        rows += (
            f"<tr><td>{html.escape(name)}</td><td>{rate}</td><td>{lo * 100:.0f}-{hi * 100:.0f}%</td>"
            f"<td>{s['pass']}/{n}</td><td>{s['uncertain']}</td></tr>"
        )
    cards = ""
    for r in run["results"]:
        if "image_path" not in r:
            cards += f"<div class=c><b>{html.escape(r['task_id'])} #{r['sample']}</b><br><i>{html.escape(str(r.get('generation_failed') or r.get('error')))}</i></div>"
            continue
        badges = "".join(
            f"<span class='b {'ok' if c['passed'] else 'unk' if c['passed'] is None else 'bad'}' title='{html.escape(json.dumps(c['detail']))}'>{html.escape(c['name'])}</span>"
            for c in r["checks"]
        )
        cards += f"<div class=c><img src='{html.escape(r['image_path'])}'><br><b>{html.escape(r['task_id'])} #{r['sample']}</b><br>{badges}</div>"
    return f"""<!doctype html><meta charset=utf-8><title>bloom-eval {html.escape(run['run_id'])}</title>
<style>body{{font:14px system-ui;margin:24px;max-width:1100px}}table{{border-collapse:collapse}}td,th{{border:1px solid #ccc;padding:4px 10px;text-align:left}}
.c{{display:inline-block;width:260px;vertical-align:top;margin:8px}}img{{width:100%;border:1px solid #ddd}}
.b{{font-size:11px;padding:1px 6px;border-radius:8px;margin-right:3px}}.ok{{background:#d6f5dd}}.bad{{background:#fbd5d5}}.unk{{background:#eee}}</style>
<h2>bloom-eval run {html.escape(run['run_id'])}</h2>
<p>brand <code>{html.escape(str(run.get('brand_id')))}</code> &middot; skill <code>{html.escape(str(run.get('skill', {}).get('skill_id')))}</code> &middot; generation failures: {gen_failures(run)}</p>
<table><tr><th>check<th>pass rate<th>95% CI<th>pass/decided<th>uncertain</tr>{rows}</table><h3>Samples</h3>{cards}"""
