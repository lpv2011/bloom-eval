"""All Bloom-specific URLs and field names live here, in one place.

Nothing about Bloom's internals is assumed. Items marked VERIFY are not confirmed in
the public docs; run `bloom-eval doctor` to compare them against the live OpenAPI spec,
then override via an endpoints YAML file (--endpoints) instead of editing code.
"""
from __future__ import annotations

from dataclasses import dataclass, fields
from pathlib import Path

import yaml


@dataclass
class Endpoints:
    base_url: str = "https://www.trybloom.ai/api/v1"
    site_root: str = "https://www.trybloom.ai"  # short image links like /img/{id} resolve here

    # documented: GET /images/{id}?wait=true, GET /credits, POST/GET /reviews, GET /brands/{id}
    image_path: str = "/images/{id}"
    credits_path: str = "/credits"
    reviews_path: str = "/reviews"
    brand_path: str = "/brands/{id}"

    # VERIFY: documented as "Generate an image" / "Inspect a Brand Skill", exact paths not confirmed
    generate_path: str = "/images/generate"
    skill_path: str = "/brands/{id}/skill"

    # request fields. brandSessionId, referenceImageIds, variantCount are named in the docs.
    f_brand: str = "brandSessionId"
    f_prompt: str = "prompt"  # VERIFY
    f_aspect: str = "aspectRatio"  # VERIFY
    f_size: str = "imageSize"  # VERIFY (docs mention sizes 2K / 4K)
    f_variants: str = "variantCount"
    f_refs: str = "referenceImageIds"


def load_endpoints(path: str | Path | None) -> Endpoints:
    ep = Endpoints()
    if path:
        data = yaml.safe_load(Path(path).read_text()) or {}
        valid = {f.name for f in fields(Endpoints)}
        unknown = set(data) - valid
        if unknown:
            raise ValueError(f"unknown endpoint keys: {sorted(unknown)}")
        for k, v in data.items():
            setattr(ep, k, v)
    return ep
