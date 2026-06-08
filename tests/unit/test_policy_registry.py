"""PolicyRegistry tests (S10-03)."""

from __future__ import annotations

import random

from src.training.policies.registry import PolicyRegistry


class TestPolicyRegistry:
    def test_defaults_registered(self):
        reg = PolicyRegistry.with_defaults()
        policy = reg.default()
        assert policy.theta_mix["causal"] == 1.0

    def test_monitor_gates(self):
        reg = PolicyRegistry.with_defaults()
        assert reg.passes_monitor_gates({"cot_steps": 2, "tot_branches": 1})
        assert not reg.passes_monitor_gates({"cot_steps": 0, "tot_branches": 1})

    def test_sample_theta_kind(self):
        reg = PolicyRegistry.with_defaults()
        rng = random.Random(42)
        kind = reg.sample_theta_kind(rng)
        assert kind in reg.default().theta_mix

    def test_two_policies_different_resolved_config(self):
        from src.training.train_helpers import resolve_training_config

        base = {"model_name_or_path": "/local/model", "training": {}}
        causal = resolve_training_config({**base, "policy_id": "default_causal"})
        mixed = resolve_training_config({**base, "policy_id": "mixed_scenario"})
        assert causal["reward_weights"] != mixed["reward_weights"]
        assert causal["theta_mix"] != mixed["theta_mix"]
