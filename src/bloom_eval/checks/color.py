"""Palette adherence using CIEDE2000.

Question answered: what fraction of the image's pixels are far from every colour
in the brand palette? Near-neutral pixels (white/grey/black) are ignored by default
because photos and text legitimately contain them.
"""
from __future__ import annotations

import numpy as np
from PIL import Image

from ..models import CheckResult

_M = np.array(
    [
        [0.4124564, 0.3575761, 0.1804375],
        [0.2126729, 0.7151522, 0.0721750],
        [0.0193339, 0.1191920, 0.9503041],
    ]
)
_WHITE = np.array([0.95047, 1.0, 1.08883])  # D65


def hex_to_rgb(h: str) -> tuple[int, int, int]:
    h = h.strip().lstrip("#")
    if len(h) == 3:
        h = "".join(c * 2 for c in h)
    if len(h) != 6:
        raise ValueError(f"bad hex colour: {h!r}")
    return int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16)


def srgb_to_lab(rgb: np.ndarray) -> np.ndarray:
    """rgb: (..., 3) floats in 0..255 -> Lab (..., 3), D65."""
    c = np.asarray(rgb, dtype=np.float64) / 255.0
    c = np.where(c <= 0.04045, c / 12.92, ((c + 0.055) / 1.055) ** 2.4)
    xyz = c @ _M.T
    t = xyz / _WHITE
    f = np.where(t > 216 / 24389, np.cbrt(t), (24389 / 27 * t + 16) / 116)
    L = 116 * f[..., 1] - 16
    a = 500 * (f[..., 0] - f[..., 1])
    b = 200 * (f[..., 1] - f[..., 2])
    return np.stack([L, a, b], axis=-1)


def ciede2000(lab1: np.ndarray, lab2: np.ndarray) -> np.ndarray:
    """Vectorised CIEDE2000 (Sharma et al. 2005). Inputs broadcast on leading dims."""
    L1, a1, b1 = lab1[..., 0], lab1[..., 1], lab1[..., 2]
    L2, a2, b2 = lab2[..., 0], lab2[..., 1], lab2[..., 2]
    C1, C2 = np.hypot(a1, b1), np.hypot(a2, b2)
    Cbar = (C1 + C2) / 2
    G = 0.5 * (1 - np.sqrt(Cbar**7 / (Cbar**7 + 25**7)))
    a1p, a2p = (1 + G) * a1, (1 + G) * a2
    C1p, C2p = np.hypot(a1p, b1), np.hypot(a2p, b2)
    h1p = np.degrees(np.arctan2(b1, a1p)) % 360
    h2p = np.degrees(np.arctan2(b2, a2p)) % 360

    dLp, dCp = L2 - L1, C2p - C1p
    dhp = h2p - h1p
    dhp = np.where(dhp > 180, dhp - 360, dhp)
    dhp = np.where(dhp < -180, dhp + 360, dhp)
    zero = (C1p * C2p) == 0
    dhp = np.where(zero, 0.0, dhp)
    dHp = 2 * np.sqrt(C1p * C2p) * np.sin(np.radians(dhp) / 2)

    Lbp, Cbp = (L1 + L2) / 2, (C1p + C2p) / 2
    hsum = h1p + h2p
    hbp = np.where(
        np.abs(h1p - h2p) <= 180,
        hsum / 2,
        np.where(hsum < 360, (hsum + 360) / 2, (hsum - 360) / 2),
    )
    hbp = np.where(zero, hsum, hbp)

    T = (
        1
        - 0.17 * np.cos(np.radians(hbp - 30))
        + 0.24 * np.cos(np.radians(2 * hbp))
        + 0.32 * np.cos(np.radians(3 * hbp + 6))
        - 0.20 * np.cos(np.radians(4 * hbp - 63))
    )
    dTheta = 30 * np.exp(-(((hbp - 275) / 25) ** 2))
    Rc = 2 * np.sqrt(Cbp**7 / (Cbp**7 + 25**7))
    Sl = 1 + 0.015 * (Lbp - 50) ** 2 / np.sqrt(20 + (Lbp - 50) ** 2)
    Sc = 1 + 0.045 * Cbp
    Sh = 1 + 0.015 * Cbp * T
    Rt = -np.sin(np.radians(2 * dTheta)) * Rc
    return np.sqrt(
        (dLp / Sl) ** 2 + (dCp / Sc) ** 2 + (dHp / Sh) ** 2 + Rt * (dCp / Sc) * (dHp / Sh)
    )


def check_color(
    img: Image.Image,
    palette: list[str],
    *,
    de_threshold: float = 12.0,
    max_off_palette: float = 0.35,
    min_coverage: float = 0.25,
    min_share: float = 0.01,
    allow_neutrals: bool = True,
    neutral_chroma: float = 12.0,
    max_side: int = 160,
) -> CheckResult:
    if not palette:
        return CheckResult("color", None, 0.0, {"error": "no palette available"})

    im = img.convert("RGB")
    im.thumbnail((max_side, max_side))
    px = srgb_to_lab(np.asarray(im, dtype=np.float64).reshape(-1, 3))
    pal = srgb_to_lab(np.array([hex_to_rgb(h) for h in palette], dtype=np.float64))

    de = ciede2000(px[:, None, :], pal[None, :, :])  # (N, P)
    nearest = de.min(axis=1)
    off = nearest > de_threshold
    if allow_neutrals:
        off &= np.hypot(px[:, 1], px[:, 2]) > neutral_chroma
    ratio = float(off.mean())

    # coverage: share of palette colours that actually appear (>= min_share of pixels)
    present = (de <= de_threshold).mean(axis=0) >= min_share
    coverage = float(present.mean())

    passed = ratio <= max_off_palette and coverage >= min_coverage
    return CheckResult(
        "color",
        passed,
        score=min(1.0 - ratio, coverage / max(min_coverage, 1e-9)),
        detail={
            "off_palette_ratio": round(ratio, 4),
            "max_off_palette": max_off_palette,
            "palette_coverage": round(coverage, 3),
            "min_coverage": min_coverage,
            "de_threshold": de_threshold,
            "mean_nearest_de": round(float(nearest.mean()), 2),
        },
    )
