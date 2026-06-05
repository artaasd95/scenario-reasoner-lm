"""θ dict → dataclass normalization tests (S9-DP-02 / SP-BUG-03)."""

from __future__ import annotations

import pytest

from src.data.theta_normalization import normalize_theta_record
from src.scenarios.causal.taxonomy import CausalTheta
from src.search.game_theta import GameTheoreticTheta
from src.training.causal_reward import CausalRewardFunction
from src.training.reward import normalize_theta


class TestThetaNormalization:
    def test_causal_dict_to_dataclass(self):
        data = {
            "chain_length": 3,
            "intervention_type": "direct",
            "domain": "physical",
            "difficulty": "easy",
        }
        result = normalize_theta_record(data)
        assert result.ok
        assert isinstance(result.theta, CausalTheta)

    def test_game_dict_to_dataclass(self):
        theta = GameTheoreticTheta.default(num_stages=2)
        result = normalize_theta_record(theta.to_dict(), kind="game")
        assert result.ok
        assert isinstance(result.theta, GameTheoreticTheta)

    def test_invalid_theta_rejected(self):
        result = normalize_theta_record({"action_dim": 3, "num_stages": 2, "action_vector": [0.0]})
        assert not result.ok
        assert result.reason

    def test_reward_accepts_dict_theta(self):
        fn = CausalRewardFunction()
        theta_dict = {
            "chain_length": 2,
            "intervention_type": "direct",
            "domain": "physical",
            "difficulty": "easy",
        }
        score = fn.score("p", "Step 1: a\nTherefore b.", "b", theta=theta_dict)
        assert 0.0 <= score <= 1.0

    def test_normalize_theta_tuple(self):
        normalized, err = normalize_theta(
            {"chain_length": 2, "intervention_type": "direct", "domain": "physical", "difficulty": "easy"}
        )
        assert err is None
        assert isinstance(normalized, CausalTheta)
