"""
Game-theoretic scenario parameters (S6).

Embeds staged decision-making via action vectors (e.g. dim 10 per stage).
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional

from src.search.manifold import ActionManifold, ManifoldKind


class InteractionMode(str, Enum):
    SINGLE_AGENT = "single_agent"
    TWO_PLAYER_ZERO_SUM = "two_player_zero_sum"
    MULTI_AGENT = "multi_agent"


def action_index_from_vector(vector_slice: List[float], menu_size: int) -> int:
    """
    Map a projected vector slice to a discrete action index in [0, menu_size).

    Uses argmax on the slice (after padding/truncation to menu_size).
    """
    if menu_size <= 0:
        raise ValueError("menu_size must be positive")
    v = list(vector_slice[:menu_size])
    while len(v) < menu_size:
        v.append(0.0)
    return int(max(range(menu_size), key=lambda i: v[i]))


@dataclass
class GameTheoreticTheta:
    """
    Parameter vector for game-theoretic search scenarios.

    Attributes:
        action_dim: Length of each stage vector (e.g. 10).
        num_stages: Number of decision stages.
        action_vector: Flat concatenation of stage vectors (length action_dim * num_stages).
        interaction_mode: Single- or multi-agent interaction model.
        menu_size: Discrete actions per stage (<= action_dim).
        manifold_kind: Feasible set for projection before discretization.
        seed: Reproducibility for samplers.
    """

    action_dim: int = 10
    num_stages: int = 3
    action_vector: List[float] = field(default_factory=list)
    interaction_mode: InteractionMode = InteractionMode.SINGLE_AGENT
    menu_size: int = 5
    manifold_kind: ManifoldKind = ManifoldKind.BOX
    seed: int = 42

    def __post_init__(self) -> None:
        if self.action_dim < 1:
            raise ValueError("action_dim must be >= 1")
        if self.num_stages < 1:
            raise ValueError("num_stages must be >= 1")
        expected = self.action_dim * self.num_stages
        if not self.action_vector:
            self.action_vector = [0.0] * expected
        if len(self.action_vector) != expected:
            raise ValueError(
                f"action_vector length must be {expected}, got {len(self.action_vector)}"
            )
        if self.menu_size > self.action_dim:
            raise ValueError("menu_size cannot exceed action_dim")

    @property
    def manifold(self) -> ActionManifold:
        return ActionManifold(kind=self.manifold_kind, dim=self.action_dim)

    def stage_slice(self, stage: int) -> List[float]:
        """Return projected vector slice for stage ``stage``."""
        if stage < 0 or stage >= self.num_stages:
            raise IndexError(f"stage {stage} out of range [0, {self.num_stages})")
        start = stage * self.action_dim
        end = start + self.action_dim
        raw = self.action_vector[start:end]
        return self.manifold.project(raw)

    def action_index_at_stage(self, stage: int) -> int:
        """Discrete action index at ``stage`` after manifold projection."""
        return action_index_from_vector(self.stage_slice(stage), self.menu_size)

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        d["interaction_mode"] = self.interaction_mode.value
        d["manifold_kind"] = self.manifold_kind.value
        return d

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "GameTheoreticTheta":
        mode = data.get("interaction_mode", InteractionMode.SINGLE_AGENT.value)
        mk = data.get("manifold_kind", ManifoldKind.BOX.value)
        return cls(
            action_dim=int(data.get("action_dim", 10)),
            num_stages=int(data.get("num_stages", 3)),
            action_vector=list(data.get("action_vector", [])),
            interaction_mode=InteractionMode(str(mode)),
            menu_size=int(data.get("menu_size", 5)),
            manifold_kind=ManifoldKind(str(mk)),
            seed=int(data.get("seed", 42)),
        )

    @classmethod
    def default(cls, action_dim: int = 10, num_stages: int = 3) -> "GameTheoreticTheta":
        """Uniform box-feasible vector for smoke tests."""
        n = action_dim * num_stages
        vec = [1.0 / action_dim] * n
        menu = min(5, action_dim)
        return cls(
            action_dim=action_dim,
            num_stages=num_stages,
            action_vector=vec,
            menu_size=menu,
        )
