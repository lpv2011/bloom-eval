"""Copy check: is the requested text rendered (nearly) exactly? Uses Tesseract OCR."""
from __future__ import annotations

import re
from difflib import SequenceMatcher

from PIL import Image

from ..models import CheckResult


def _norm(s: str) -> str:
    return " ".join(re.sub(r"[^\w\s']", " ", s.lower()).split())


def best_window_ratio(expected: str, observed: str) -> float:
    """Best similarity between `expected` and any same-length word window of `observed`."""
    e, o = _norm(expected), _norm(observed)
    if not e or not o:
        return 0.0
    if e in o:
        return 1.0
    n = len(e.split())
    words = o.split()
    best = 0.0
    for size in {max(1, n - 1), n, n + 1}:
        for i in range(0, max(1, len(words) - size + 1)):
            best = max(best, SequenceMatcher(None, e, " ".join(words[i : i + size])).ratio())
    return best


def check_text(img: Image.Image, expected: str | None, *, min_ratio: float = 0.95) -> CheckResult:
    if not expected:
        return CheckResult("text", None, 0.0, {"error": "no expected_text given"})
    try:
        import pytesseract
    except ImportError:
        return CheckResult("text", None, 0.0, {"error": "pytesseract not installed (pip install .[vision])"})

    try:
        g = img.convert("L")
        # OCR is much more reliable on upscaled images
        if max(g.size) < 1600:
            f = 1600 / max(g.size)
            g = g.resize((int(g.width * f), int(g.height * f)))
        observed = pytesseract.image_to_string(g, config="--psm 11")
    except Exception as exc:  # tesseract binary missing, etc.
        return CheckResult("text", None, 0.0, {"error": f"ocr failed: {exc}"})

    ratio = best_window_ratio(expected, observed)
    return CheckResult(
        "text",
        ratio >= min_ratio,
        score=ratio,
        detail={"expected": expected, "ocr": _norm(observed)[:300], "min_ratio": min_ratio},
    )
