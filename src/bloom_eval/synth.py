"""Synthetic brand banners + fault injection.

Used by `bloom-eval validate` to measure the checkers themselves: we know exactly what is
wrong with each image, so we can compute how often each check catches it (recall) and how
often it wrongly flags a clean image (false-positive rate). No Bloom account needed.
"""
from __future__ import annotations

import colorsys
import random
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont

from .checks.color import hex_to_rgb

_FONT_CANDIDATES = [
    "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
    "/Library/Fonts/Arial Bold.ttf",
    "C:/Windows/Fonts/arialbd.ttf",
]


def font(size: int) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    for p in _FONT_CANDIDATES:
        if Path(p).exists():
            return ImageFont.truetype(p, size)
    return ImageFont.load_default(size=size)


def make_logo(fg: str, bg: str, size: int = 320, seed: int = 7) -> Image.Image:
    """A deliberately feature-rich synthetic mark (so ORB has corners to latch onto)."""
    rng = random.Random(seed)
    im = Image.new("RGB", (size, size), bg)
    d = ImageDraw.Draw(im)
    d.ellipse([8, 8, size - 8, size - 8], outline=fg, width=14)
    for _ in range(7):
        pts = [(rng.randint(40, size - 40), rng.randint(40, size - 40)) for _ in range(3)]
        d.polygon(pts, fill=fg if rng.random() > 0.4 else bg, outline=fg)
    d.text((size // 2 - 62, size // 2 - 30), "BLM", font=font(64), fill=fg)
    return im


def make_banner(
    palette: list[str],
    logo: Image.Image,
    headline: str,
    *,
    size: tuple[int, int] = (1200, 630),
    logo_stretch: tuple[float, float] = (1.0, 1.0),
    include_logo: bool = True,
    seed: int = 0,
) -> Image.Image:
    rng = random.Random(seed)
    bg, fg = hex_to_rgb(palette[1]), hex_to_rgb(palette[0])
    im = Image.new("RGB", size, bg)
    d = ImageDraw.Draw(im)
    d.rectangle([0, size[1] - 90, size[0], size[1]], fill=fg)
    d.text((70 + rng.randint(0, 40), 200 + rng.randint(0, 60)), headline, font=font(84), fill=fg)
    if include_logo:
        w, h = int(190 * logo_stretch[0]), int(190 * logo_stretch[1])
        lg = logo.resize((w, h), Image.LANCZOS)
        im.paste(lg, (size[0] - w - 60 - rng.randint(0, 30), 50 + rng.randint(0, 30)))
    return im


def hue_shift(im: Image.Image, degrees: float) -> Image.Image:
    hsv = np.asarray(im.convert("HSV")).copy()
    hsv[..., 0] = ((hsv[..., 0].astype(int) + int(degrees / 360 * 255)) % 256).astype(np.uint8)
    return Image.fromarray(hsv, "HSV").convert("RGB")


def typo(text: str, rng: random.Random) -> str:
    """Transpose two different adjacent letters in up to two words (a realistic generative-model slip).

    Guaranteed to change the string, so a 'typo' sample is never secretly clean.
    """
    words = text.split()
    order = list(range(len(words)))
    rng.shuffle(order)
    changed = 0
    for i in order:
        w = words[i]
        spots = [j for j in range(1, len(w) - 2) if w[j] != w[j + 1]]
        if spots:
            j = rng.choice(spots)
            words[i] = w[:j] + w[j + 1] + w[j] + w[j + 2 :]
            changed += 1
            if changed == 2:
                break
    return " ".join(words)


# fault name -> (builder(palette, logo, headline, seed) -> image, checks that SHOULD fail)
def build_faults(palette: list[str], logo: Image.Image, headline: str):
    def clean(seed):
        return make_banner(palette, logo, headline, seed=seed)

    return {
        "hue_shift": (lambda s: hue_shift(clean(s), 120), {"color"}),
        "logo_stretched": (lambda s: make_banner(palette, logo, headline, logo_stretch=(1.7, 1.0), seed=s), {"logo"}),
        "logo_missing": (lambda s: make_banner(palette, logo, headline, include_logo=False, seed=s), {"logo"}),
        "headline_typo": (
            lambda s: make_banner(palette, logo, typo(headline, random.Random(s)), seed=s),
            {"text"},
        ),
    }, clean
