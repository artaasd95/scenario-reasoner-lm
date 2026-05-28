"""
Market-making *reasoning* parameters (S6).

Search over reasoning templates and strategies — not live order execution.
Pairs with ``GameTheoreticTheta`` for staged action vectors in future fixtures.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List

from src.search.game_theta import GameTheoreticTheta


@dataclass
class MarketMakingReasoningTheta:
    """
    θ for searching good reasoning approaches in market-making settings.

    Attributes:
        spread_regime: tight | normal | wide
        inventory_pressure: low | medium | high
        reasoning_strategy_pool: Named templates to search (e.g. adverse_selection).
        search_budget: Max expansions / rollouts for the reasoning search.
        game: Optional staged action vector parameters.
    """

    spread_regime: str = "normal"
    inventory_pressure: str = "medium"
    reasoning_strategy_pool: List[str] = field(
        default_factory=lambda: [
            "inventory_skew",
            "adverse_selection",
            "flow_toxicity",
        ]
    )
    search_budget: int = 32
    game: GameTheoreticTheta = field(default_factory=GameTheoreticTheta.default)
    seed: int = 42

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        d["game"] = self.game.to_dict()
        return d

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "MarketMakingReasoningTheta":
        game_data = data.get("game")
        game = (
            GameTheoreticTheta.from_dict(game_data)
            if isinstance(game_data, dict)
            else GameTheoreticTheta.default()
        )
        return cls(
            spread_regime=str(data.get("spread_regime", "normal")),
            inventory_pressure=str(data.get("inventory_pressure", "medium")),
            reasoning_strategy_pool=list(
                data.get(
                    "reasoning_strategy_pool",
                    ["inventory_skew", "adverse_selection", "flow_toxicity"],
                )
            ),
            search_budget=int(data.get("search_budget", 32)),
            game=game,
            seed=int(data.get("seed", 42)),
        )
