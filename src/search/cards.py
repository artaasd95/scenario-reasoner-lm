"""
Composable search operators as serializable cards (S6).

Node / Action / Algorithm cards let search policies traverse structured spaces
with the same auditability as enterprise scenario output cards.
"""

from __future__ import annotations

import uuid
from dataclasses import asdict, dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional


class SearchOperator(str, Enum):
    """Named search operators algorithms may expose as cards."""

    EXPAND = "expand"
    ROLLOUT = "rollout"
    BACKPROP = "backprop"
    PRUNE = "prune"
    RANK = "rank"
    SELECT = "select"


@dataclass
class NodeCard:
    """
    A vertex in the search graph (reasoning state at depth d).

    Attributes:
        node_id: Stable id within a search run.
        state_summary: Short natural-language or structured state label.
        depth: Depth from root (0 = initial).
        parent_id: Parent node id, or None for root.
        theta_slice: Optional θ fragment active at this node.
        metadata: Extra fields (domain, fixture id, span id).
    """

    node_id: str
    state_summary: str
    depth: int
    parent_id: Optional[str] = None
    theta_slice: Optional[Dict[str, Any]] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    @classmethod
    def new(
        cls,
        state_summary: str,
        depth: int,
        parent_id: Optional[str] = None,
        **kwargs: Any,
    ) -> "NodeCard":
        return cls(
            node_id=str(uuid.uuid4()),
            state_summary=state_summary,
            depth=depth,
            parent_id=parent_id,
            **kwargs,
        )

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "NodeCard":
        return cls(**data)


@dataclass
class ActionCard:
    """
    A decision at a node, optionally derived from a game-theoretic vector slice.

    Attributes:
        action_id: Unique id for this decision event.
        node_id: Node where the action was taken.
        stage: Stage index (0 .. num_stages-1).
        vector_slice: Sub-vector used to pick the action (e.g. length K).
        label: Human-readable action name.
        player_id: 0 for single-agent; >0 for multi-agent extensions.
        discrete_index: Index into the stage action menu, if resolved.
    """

    action_id: str
    node_id: str
    stage: int
    vector_slice: List[float]
    label: str
    player_id: int = 0
    discrete_index: Optional[int] = None

    @classmethod
    def new(
        cls,
        node_id: str,
        stage: int,
        vector_slice: List[float],
        label: str,
        **kwargs: Any,
    ) -> "ActionCard":
        return cls(
            action_id=str(uuid.uuid4()),
            node_id=node_id,
            stage=stage,
            vector_slice=list(vector_slice),
            label=label,
            **kwargs,
        )

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ActionCard":
        return cls(**data)


@dataclass
class AlgorithmCard:
    """
    Describes a search algorithm or pipeline stage as an operator.

    Algorithms traverse nodes by emitting ActionCards and attaching to NodeCards.
    """

    algorithm_id: str
    name: str
    operator: SearchOperator
    config: Dict[str, Any] = field(default_factory=dict)

    @classmethod
    def new(
        cls,
        name: str,
        operator: SearchOperator,
        config: Optional[Dict[str, Any]] = None,
    ) -> "AlgorithmCard":
        return cls(
            algorithm_id=str(uuid.uuid4()),
            name=name,
            operator=operator,
            config=config or {},
        )

    def apply_to(self, node: NodeCard) -> Dict[str, Any]:
        """
        Return metadata for applying this operator to *node* (hook for S6-02 wiring).

        Does not mutate the graph; monitors and runners perform side effects.
        """
        return {
            "algorithm_id": self.algorithm_id,
            "operator": self.operator.value,
            "node_id": node.node_id,
            "depth": node.depth,
            "config": dict(self.config),
        }

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        d["operator"] = self.operator.value
        return d

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "AlgorithmCard":
        op = data.get("operator", SearchOperator.EXPAND.value)
        if isinstance(op, SearchOperator):
            operator = op
        else:
            operator = SearchOperator(str(op))
        return cls(
            algorithm_id=data["algorithm_id"],
            name=data["name"],
            operator=operator,
            config=data.get("config", {}),
        )
