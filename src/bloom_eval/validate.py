"""Measure the checkers themselves with injected faults (no Bloom access needed)."""
from __future__ import annotations

from .checks import check_color, check_logo, check_text
from .synth import build_faults, make_logo

DEFAULT_PALETTE = ["#0B3D2E", "#F4E9D8"]
DEFAULT_HEADLINE = "Meet your morning"


def run_checks(img, palette, logo, headline):
    return {
        "color": check_color(img, palette),
        "logo": check_logo(img, logo),
        "text": check_text(img, headline),
    }


def validate(n: int = 12, palette=None, headline: str = DEFAULT_HEADLINE) -> dict:
    palette = palette or DEFAULT_PALETTE
    logo = make_logo(palette[0], palette[1])
    faults, clean = build_faults(palette, logo, headline)

    stats = {c: {"clean_pass": 0, "clean_fail": 0, "clean_unc": 0} for c in ("color", "logo", "text")}
    for s in range(n):
        for name, r in run_checks(clean(s), palette, logo, headline).items():
            key = "clean_unc" if r.passed is None else "clean_pass" if r.passed else "clean_fail"
            stats[name][key] += 1

    fault_rows = {}
    for fname, (build, targets) in faults.items():
        caught = {t: 0 for t in targets}
        unc = {t: 0 for t in targets}
        for s in range(n):
            res = run_checks(build(s), palette, logo, headline)
            for t in targets:
                if res[t].passed is None:
                    unc[t] += 1
                elif res[t].passed is False:
                    caught[t] += 1
        fault_rows[fname] = {t: {"caught": caught[t], "uncertain": unc[t], "n": n} for t in targets}
    return {"n": n, "clean": stats, "faults": fault_rows}


def print_summary(res: dict) -> None:
    n = res["n"]
    print(f"\nChecker validation on synthetic banners (n={n} per condition)\n")
    print("False positives on CLEAN images (a good check passes them):")
    for c, s in res["clean"].items():
        print(f"  {c:6s} pass {s['clean_pass']:>3}/{n}   wrongly failed {s['clean_fail']:>3}   uncertain {s['clean_unc']:>3}")
    print("\nRecall on injected faults (a good check fails them):")
    for f, row in res["faults"].items():
        for c, v in row.items():
            print(f"  {f:15s} -> {c:6s} caught {v['caught']:>3}/{v['n']}   uncertain {v['uncertain']:>3}")
    print()
