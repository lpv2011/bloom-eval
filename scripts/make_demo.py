"""Create demo/ with a good baseline set, a "bad brand edit" candidate set, and a suite.

    python scripts/make_demo.py
    bloom-eval score demo/suite.yaml demo/good --out demo/runs
    bloom-eval score demo/suite.yaml demo/bad  --out demo/runs
    bloom-eval compare demo/runs/<first> demo/runs/<second>     # exits 1: regression
"""
from pathlib import Path

from bloom_eval.synth import hue_shift, make_banner, make_logo

PAL = ["#0B3D2E", "#F4E9D8"]
root = Path("demo")
for d in ("good", "bad", "assets"):
    (root / d).mkdir(parents=True, exist_ok=True)

logo = make_logo(*PAL)
logo.save(root / "assets/logo.png")
for i in range(6):
    make_banner(PAL, logo, "Meet your morning", seed=i).save(root / f"good/hero_{i}.png")
    bad = make_banner(PAL, logo, "Meet your morning", logo_stretch=(1.7, 1.0), seed=i)
    (hue_shift(bad, 120) if i % 2 else bad).save(root / f"bad/hero_{i}.png")

(root / "suite.yaml").write_text(
    'brand: { id: demo, palette: ["#0B3D2E", "#F4E9D8"], logo: assets/logo.png }\n'
    "checks: [color, text, logo]\n"
    'tasks:\n  - { id: hero, prompt: "-", expected_text: "Meet your morning" }\n'
)
print("demo/ ready")
