"""
Search-layer abstractions (S6): game-theoretic θ, algorithm cards, graph monitoring.

Complements ``src.scenarios`` (scenario instances) and ``src.monitoring`` (CoT/ToT/aha).
"""

from src.search.cards import ActionCard, AlgorithmCard, NodeCard, SearchOperator
from src.search.game_theta import GameTheoreticTheta, InteractionMode, action_index_from_vector
from src.search.graph_monitor import SearchGraphMonitor, SearchGraphReport
from src.search.manifold import ActionManifold, ManifoldKind

__all__ = [
    "ActionCard",
    "AlgorithmCard",
    "NodeCard",
    "SearchOperator",
    "GameTheoreticTheta",
    "InteractionMode",
    "action_index_from_vector",
    "SearchGraphMonitor",
    "SearchGraphReport",
    "ActionManifold",
    "ManifoldKind",
]
