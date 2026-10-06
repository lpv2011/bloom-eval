from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from . import __version__


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(prog="bloom-eval", description="Brand CI for Bloom Brand Skills")
    ap.add_argument("--version", action="version", version=__version__)
    ap.add_argument("--endpoints", help="YAML overriding Bloom endpoint paths/field names (see config.py)")
    sub = ap.add_subparsers(dest="cmd", required=True)

    p = sub.add_parser("doctor", help="check configured endpoints against Bloom's OpenAPI spec")
    p.add_argument("--spec", help="local spec.json instead of fetching")

    p = sub.add_parser("validate", help="measure the checkers on synthetic images with injected faults (offline)")
    p.add_argument("-n", type=int, default=12)
    p.add_argument("--json", help="write results here")

    p = sub.add_parser("run", help="generate with Bloom and score (uses credits!)")
    p.add_argument("suite")
    p.add_argument("--out", default="runs")
    p.add_argument("--with-review", action="store_true", help="also request Bloom Reviews (private beta)")
    p.add_argument("--workers", type=int, default=4)
    p.add_argument("--yes", action="store_true", help="skip the credit confirmation")

    p = sub.add_parser("score", help="score a folder of existing images named <task_id>_<n>.png")
    p.add_argument("suite")
    p.add_argument("folder")
    p.add_argument("--out", default="runs")

    p = sub.add_parser("report", help="render report.html for a run")
    p.add_argument("run")

    p = sub.add_parser("compare", help="compare candidate run against baseline; exit 1 on regression")
    p.add_argument("baseline")
    p.add_argument("candidate")
    p.add_argument("--alpha", type=float, default=0.05)
    p.add_argument("--min-drop", type=float, default=0.05)

    args = ap.parse_args(argv)

    from .config import load_endpoints

    ep = load_endpoints(args.endpoints)

    if args.cmd == "doctor":
        from .doctor import run_doctor

        sys.exit(run_doctor(ep, args.spec))

    if args.cmd == "validate":
        from .validate import print_summary, validate

        res = validate(args.n)
        print_summary(res)
        if args.json:
            Path(args.json).write_text(json.dumps(res, indent=2))
        return

    if args.cmd == "run":
        from .client import BloomClient
        from .runner import load_suite, run_suite

        suite = load_suite(args.suite)
        n = len(suite["tasks"]) * suite["samples"]
        per = 2 if str(suite.get("size", "")).upper() == "4K" else 1
        client = BloomClient(ep)
        print(f"{n} generations ~= {n * per} credits (plus edits/reviews if any). Balance: {client.credits()}")
        if not args.yes and input("Proceed? [y/N] ").strip().lower() != "y":
            sys.exit("aborted")
        out = run_suite(client, suite, args.out, with_review=args.with_review, workers=args.workers)
        _finish(out)
        return

    if args.cmd == "score":
        from .runner import load_suite, score_folder

        out = score_folder(args.folder, load_suite(args.suite), args.out)
        _finish(out)
        return

    from .report import compare, format_compare, load_run, render_html

    if args.cmd == "report":
        run = load_run(args.run)
        out = Path(run["_dir"]) / "report.html"
        out.write_text(render_html(run))
        print(out)
        return

    if args.cmd == "compare":
        a, b = load_run(args.baseline), load_run(args.candidate)
        rows = compare(a, b, alpha=args.alpha, min_drop=args.min_drop)
        print(format_compare(rows, a, b))
        sys.exit(1 if any(r["regression"] for r in rows) else 0)


def _finish(out: Path) -> None:
    from .report import aggregate, load_run, render_html

    run = load_run(out)
    (out / "report.html").write_text(render_html(run))
    print(f"\nrun written to {out}")
    for name, s in sorted(aggregate(run).items()):
        n = s["pass"] + s["fail"]
        print(f"  {name:20s} {s['pass']}/{n} pass   ({s['uncertain']} uncertain)")
    print(f"  report: {out / 'report.html'}")


if __name__ == "__main__":
    main()
