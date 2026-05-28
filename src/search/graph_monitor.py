"""
Per-node search graph monitoring (S6).

Tracks visits, expansions, and prunes during tree / policy-guided search.
Complements S5 ``reasoning_path_audit`` (pipeline span order).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from src.search.cards import NodeCard


@dataclass
class NodeStats:
    node_id: str
    depth: int
    visit_count: int = 0
    expansion_count: int = 0
    prune_count: int = 0
    prune_reasons: List[str] = field(default_factory=list)
    algorithm_ids: List[str] = field(default_factory=list)
    span_ids: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "node_id": self.node_id,
            "depth": self.depth,
            "visit_count": self.visit_count,
            "expansion_count": self.expansion_count,
            "prune_count": self.prune_count,
            "prune_reasons": list(self.prune_reasons),
            "algorithm_ids": list(self.algorithm_ids),
            "span_ids": list(self.span_ids),
        }


@dataclass
class SearchGraphReport:
    nodes: List[NodeStats]
    total_visits: int
    total_expansions: int
    total_prunes: int
    branching_factor_estimate: float
    max_depth: int

    def to_dict(self) -> Dict[str, Any]:
        return {
            "nodes": [n.to_dict() for n in self.nodes],
            "total_visits": self.total_visits,
            "total_expansions": self.total_expansions,
            "total_prunes": self.total_prunes,
            "branching_factor_estimate": self.branching_factor_estimate,
            "max_depth": self.max_depth,
        }


class SearchGraphMonitor:
    """In-memory monitor for search algorithms traversing NodeCards."""

    def __init__(self) -> None:
        self._stats: Dict[str, NodeStats] = {}

    def _get_or_create(self, node: NodeCard) -> NodeStats:
        if node.node_id not in self._stats:
            self._stats[node.node_id] = NodeStats(
                node_id=node.node_id,
                depth=node.depth,
            )
        return self._stats[node.node_id]

    def record_visit(
        self,
        node: NodeCard,
        algorithm_id: str,
        span_id: Optional[str] = None,
    ) -> None:
        stats = self._get_or_create(node)
        stats.visit_count += 1
        if algorithm_id and algorithm_id not in stats.algorithm_ids:
            stats.algorithm_ids.append(algorithm_id)
        if span_id and span_id not in stats.span_ids:
            stats.span_ids.append(span_id)

    def record_expansion(
        self,
        node: NodeCard,
        algorithm_id: str,
        span_id: Optional[str] = None,
    ) -> None:
        stats = self._get_or_create(node)
        stats.expansion_count += 1
        self.record_visit(node, algorithm_id, span_id=span_id)

    def record_prune(
        self,
        node: NodeCard,
        reason: str,
        algorithm_id: str = "",
    ) -> None:
        stats = self._get_or_create(node)
        stats.prune_count += 1
        stats.prune_reasons.append(reason)
        if algorithm_id:
            self.record_visit(node, algorithm_id)

    def snapshot(self) -> SearchGraphReport:
        nodes = sorted(self._stats.values(), key=lambda n: (n.depth, n.node_id))
        total_visits = sum(n.visit_count for n in nodes)
        total_expansions = sum(n.expansion_count for n in nodes)
        total_prunes = sum(n.prune_count for n in nodes)
        max_depth = max((n.depth for n in nodes), default=0)

        # Rough estimate: expansions / nodes with at least one expansion
        expanded = [n for n in nodes if n.expansion_count > 0]
        if expanded:
            bf = sum(n.expansion_count for n in expanded) / len(expanded)
        else:
            bf = 0.0

        return SearchGraphReport(
            nodes=nodes,
            total_visits=total_visits,
            total_expansions=total_expansions,
            total_prunes=total_prunes,
            branching_factor_estimate=bf,
            max_depth=max_depth,
        )

    def reset(self) -> None:
        self._stats.clear()
