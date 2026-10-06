"""Logo presence + integrity via ORB feature matching and an affine fit.

Integrity = the logo is found AND its shape is not stretched: we fit an affine map
logo -> image and require its two singular values to be close (uniform scaling).
This is a heuristic. It works for logos with enough texture/corners; flat wordmarks
may return `uncertain` (too few keypoints).
"""
from __future__ import annotations

import numpy as np
from PIL import Image

from ..models import CheckResult


def _gray(im: Image.Image, min_side: int = 0) -> np.ndarray:
    import cv2

    if im.mode in ("RGBA", "LA", "P"):
        im = im.convert("RGBA")
        bg = Image.new("RGBA", im.size, (255, 255, 255, 255))
        im = Image.alpha_composite(bg, im)
    im = im.convert("RGB")
    if min_side and min(im.size) < min_side:
        f = min_side / min(im.size)
        im = im.resize((int(im.width * f), int(im.height * f)))
    return cv2.cvtColor(np.asarray(im), cv2.COLOR_RGB2GRAY)


def check_logo(
    img: Image.Image,
    logo: Image.Image | None,
    *,
    min_inliers: int = 10,
    max_distortion: float = 1.15,
) -> CheckResult:
    if logo is None:
        return CheckResult("logo", None, 0.0, {"error": "no logo reference given"})
    try:
        import cv2
    except ImportError:
        return CheckResult("logo", None, 0.0, {"error": "opencv not installed (pip install .[vision])"})

    g_logo, g_img = _gray(logo, min_side=256), _gray(img)
    orb = cv2.ORB_create(nfeatures=3000)
    k1, d1 = orb.detectAndCompute(g_logo, None)
    k2, d2 = orb.detectAndCompute(g_img, None)
    if d1 is None or d2 is None or len(k1) < 8:
        return CheckResult("logo", None, 0.0, {"error": "too few keypoints in logo or image"})

    pairs = cv2.BFMatcher(cv2.NORM_HAMMING).knnMatch(d1, d2, k=2)
    good = [p[0] for p in pairs if len(p) == 2 and p[0].distance < 0.8 * p[1].distance]
    if len(good) < 4:
        return CheckResult("logo", False, 0.0, {"reason": "logo not found", "good_matches": len(good)})

    src = np.float32([k1[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
    dst = np.float32([k2[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)
    M, inl = cv2.estimateAffine2D(src, dst, ransacReprojThreshold=4.0)
    if M is None:
        return CheckResult("logo", False, 0.0, {"reason": "no consistent transform", "good_matches": len(good)})

    n_in = int(inl.sum())
    s = np.linalg.svd(M[:, :2], compute_uv=False)
    distortion = float(s[0] / max(s[1], 1e-9))
    found = n_in >= min_inliers
    intact = distortion <= max_distortion
    return CheckResult(
        "logo",
        bool(found and intact),
        score=float(min(1.0, n_in / (2 * min_inliers)) / max(1.0, distortion)),
        detail={
            "inliers": n_in,
            "min_inliers": min_inliers,
            "distortion": round(distortion, 3),
            "max_distortion": max_distortion,
            "reason": "ok" if found and intact else ("not found" if not found else "logo distorted"),
        },
    )
