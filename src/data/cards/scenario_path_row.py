"""
ScenarioPathRow — record card wrapping search path types (S9-DP-02).
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional

from src.data.theta_normalization import normalize_theta_record
from src.search.cards import ActionCard, AlgorithmCard, NodeCard


@dataclass
class ScenarioPathRow:
    """
    Tabular record for one search/scenario path.

    Wraps :class:`~src.search.cards` node/action/algorithm cards plus θ.
    """

    path_id: str
    nodes: List[NodeCard] = field(default_factory=list)
    actions: List[ActionCard] = field(default_factory=list)
    algorithms: List[AlgorithmCard] = field(default_factory=list)
    theta: Dict[str, Any] = field(default_factory=dict)
    theta_kind: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    def normalized_theta(self):
        """Return typed θ after normalize stage."""
        result = normalize_theta_record(self.theta, kind=self.theta_kind)
        if not result.ok:
            raise ValueError(result.reason)
        return result.theta

    def to_dict(self) -> Dict[str, Any]:
        return {
            "path_id": self.path_id,
            "nodes": [n.to_dict() for n in self.nodes],
            "actions": [a.to_dict() for a in self.actions],
            "algorithms": [a.to_dict() for a in self.algorithms],
            "theta": dict(self.theta),
            "theta_kind": self.theta_kind,
            "metadata": dict(self.metadata),
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ScenarioPathRow":
        return cls(
            path_id=data["path_id"],
            nodes=[NodeCard.from_dict(n) for n in data.get("nodes", [])],
            actions=[ActionCard.from_dict(a) for a in data.get("actions", [])],
            algorithms=[AlgorithmCard.from_dict(a) for a in data.get("algorithms", [])],
            theta=dict(data.get("theta", {})),
            theta_kind=data.get("theta_kind"),
            metadata=dict(data.get("metadata", {})),
        )
