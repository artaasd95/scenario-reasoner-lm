"""PolicyRegistry — θ mix, reward weights, monitor gates (S10-03)."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


@dataclass
class TrainingPolicy:
    name: str
    theta_mix: Dict[str, float] = field(default_factory=dict)
    reward_weights: Dict[str, float] = field(default_factory=lambda: {
        "alpha_cot": 0.15,
        "beta_tot": 0.10,
        "gamma_aha": 0.05,
    })
    monitor_gates: Dict[str, float] = field(default_factory=lambda: {
        "cot_min_steps": 1.0,
        "tot_min_branches": 0.0,
        "aha_min_moments": 0.0,
    })

    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "theta_mix": dict(self.theta_mix),
            "reward_weights": dict(self.reward_weights),
            "monitor_gates": dict(self.monitor_gates),
        }


class PolicyRegistry:
    def __init__(self) -> None:
        self._policies: Dict[str, TrainingPolicy] = {}

    def register(self, policy: TrainingPolicy) -> None:
        self._policies[policy.name] = policy

    def get(self, name: str) -> TrainingPolicy:
        if name not in self._policies:
            raise KeyError(f"Unknown policy: {name}")
        return self._policies[name]

    def default(self) -> TrainingPolicy:
        return self.get("causal_default")

    @classmethod
    def with_defaults(cls) -> "PolicyRegistry":
        reg = cls()
        reg.register(
            TrainingPolicy(
                name="causal_default",
                theta_mix={"causal": 1.0},
            )
        )
        reg.register(
            TrainingPolicy(
                name="mixed_scenario",
                theta_mix={"causal": 0.5, "enterprise": 0.3, "game": 0.2},
                reward_weights={"alpha_cot": 0.12, "beta_tot": 0.12, "gamma_aha": 0.06},
            )
        )
        return reg

    def sample_theta_kind(self, rng_mod: Any) -> str:
        """Sample a θ kind from the default policy mix."""
        policy = self.default()
        kinds = list(policy.theta_mix.keys())
        weights = list(policy.theta_mix.values())
        return rng_mod.choices(kinds, weights=weights, k=1)[0]

    def passes_monitor_gates(self, stats: Dict[str, float], policy_name: str = "causal_default") -> bool:
        policy = self.get(policy_name)
        gates = policy.monitor_gates
        if stats.get("cot_steps", 0) < gates.get("cot_min_steps", 0):
            return False
        if stats.get("tot_branches", 0) < gates.get("tot_min_branches", 0):
            return False
        return True
