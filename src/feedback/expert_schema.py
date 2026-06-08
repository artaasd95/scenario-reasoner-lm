"""Expert feedback schema (SR-22)."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, Optional


class ExpertAction(str, Enum):
    APPROVE = "APPROVE"
    REJECT = "REJECT"
    FLAG_NUMERICAL = "FLAG_NUMERICAL"
    FLAG_LOGIC = "FLAG_LOGIC"
    ADD_NOTE = "ADD_NOTE"


@dataclass
class ExpertFeedback:
    feedback_id: str
    scenario_id: str
    action: ExpertAction
    note: str = ""
    theta: Dict[str, Any] = field(default_factory=dict)
    status: str = "pending"

    def to_dict(self) -> Dict[str, Any]:
        return {
            "feedback_id": self.feedback_id,
            "scenario_id": self.scenario_id,
            "action": self.action.value,
            "note": self.note,
            "theta": dict(self.theta),
            "status": self.status,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ExpertFeedback":
        return cls(
            feedback_id=str(data["feedback_id"]),
            scenario_id=str(data["scenario_id"]),
            action=ExpertAction(data["action"]),
            note=str(data.get("note", "")),
            theta=dict(data.get("theta", {})),
            status=str(data.get("status", "pending")),
        )
