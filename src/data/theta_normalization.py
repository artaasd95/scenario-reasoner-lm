"""θ dict → typed dataclass at data-platform normalize stage (S9-DP-02)."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional

from src.training.reward import infer_theta_kind, normalize_theta


@dataclass
class ThetaNormalizationResult:
    theta: Optional[Any]
    kind: str
    ok: bool
    reason: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return {
            "ok": self.ok,
            "kind": self.kind,
            "reason": self.reason,
            "theta": self.theta.to_dict() if self.theta is not None and hasattr(self.theta, "to_dict") else {},
        }


def normalize_theta_record(
    data: Dict[str, Any],
    *,
    kind: Optional[str] = None,
) -> ThetaNormalizationResult:
    """Normalize a θ dict; reject invalid θ with reason."""
    theta_kind = kind or data.get("theta_kind") or infer_theta_kind(data)
    normalized, err = normalize_theta(data, kind=theta_kind)
    if err:
        return ThetaNormalizationResult(theta=None, kind=theta_kind, ok=False, reason=err)
    return ThetaNormalizationResult(theta=normalized, kind=theta_kind, ok=True)
