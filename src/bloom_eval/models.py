from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any


@dataclass
class CheckResult:
    """Outcome of one check on one image.

    passed: True / False, or None when the check could not decide ("uncertain").
    score:  0..1, higher is better (check-specific meaning, see each check).
    """

    name: str
    passed: bool | None
    score: float = 0.0
    detail: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)
