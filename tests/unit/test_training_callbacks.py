"""Training callback tests (S10-08)."""

from __future__ import annotations

from src.training.callbacks.reward_theta_callbacks import (
    RewardDecompositionCallback,
    ThetaStratifiedEarlyStopCallback,
)


class TestTrainingCallbacks:
    def test_reward_decomposition_aggregate(self):
        cb = RewardDecompositionCallback()
        cb.on_step_end({"R_task": 0.5, "R_total": 0.6})
        cb.on_step_end({"R_task": 0.7, "R_total": 0.8})
        agg = cb.aggregate()
        assert agg["R_task"] == 0.6

    def test_theta_early_stop(self):
        cb = ThetaStratifiedEarlyStopCallback(patience=2, min_delta=0.05)
        assert not cb.on_epoch_end({"easy": 0.5})
        assert not cb.on_epoch_end({"easy": 0.51})
        assert cb.on_epoch_end({"easy": 0.51})
