"""Optional vision-LLM judge for things code can't decide (typography feel, voice).

Not used by default. Needs ANTHROPIC_API_KEY. Pin the model and rubric so scores are
comparable across runs; report judge/human agreement before trusting it.
"""
from __future__ import annotations

import base64
import io
import json
import os

import httpx
from PIL import Image

from ..models import CheckResult

RUBRIC = """You are auditing ONE marketing image against written brand guidance.
Judge only what the guidance explicitly states; do not apply personal taste.
Aspect to judge: {aspect}.
Brand guidance:
---
{guidance}
---
Reply with JSON only: {{"verdict": "pass"|"fail"|"uncertain", "reason": "<=25 words"}}"""


def check_judge(img: Image.Image, aspect: str, guidance: str, *, model: str | None = None) -> CheckResult:
    key = os.environ.get("ANTHROPIC_API_KEY")
    if not key:
        return CheckResult(f"judge:{aspect}", None, 0.0, {"error": "ANTHROPIC_API_KEY not set"})
    buf = io.BytesIO()
    im = img.convert("RGB")
    im.thumbnail((1568, 1568))
    im.save(buf, "PNG")
    body = {
        "model": model or os.environ.get("BLOOM_EVAL_JUDGE_MODEL", "claude-sonnet-5-5"),
        "max_tokens": 200,
        "temperature": 0,
        "messages": [
            {
                "role": "user",
                "content": [
                    {
                        "type": "image",
                        "source": {
                            "type": "base64",
                            "media_type": "image/png",
                            "data": base64.b64encode(buf.getvalue()).decode(),
                        },
                    },
                    {"type": "text", "text": RUBRIC.format(aspect=aspect, guidance=guidance[:12000])},
                ],
            }
        ],
    }
    try:
        r = httpx.post(
            "https://api.anthropic.com/v1/messages",
            headers={"x-api-key": key, "anthropic-version": "2023-06-01"},
            json=body,
            timeout=60,
        )
        r.raise_for_status()
        text = "".join(b.get("text", "") for b in r.json()["content"])
        data = json.loads(text[text.index("{") : text.rindex("}") + 1])
    except Exception as exc:
        return CheckResult(f"judge:{aspect}", None, 0.0, {"error": str(exc)[:200]})
    v = data.get("verdict")
    return CheckResult(
        f"judge:{aspect}",
        True if v == "pass" else False if v == "fail" else None,
        1.0 if v == "pass" else 0.0,
        {"reason": data.get("reason", "")},
    )
