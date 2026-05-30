"""
Unit tests for scenario reasoning batch (S6-03 / S7-03).
"""

from __future__ import annotations

import pytest

from src.scenarios.enumerated_path_generator import generate_from_fixture, load_enumerated_fixtures
from src.scenarios.exploratory_path_generator import generate_from_exploratory_fixture, load_exploratory_fixtures
from src.scenarios.resource_gate import assert_mock_or_gated
from src.scenarios.scenario_reasoning_batch import (
    ScenarioReasoningBatch,
    run_reasoning_for_path,
)


class TestScenarioReasoningBatch:
    def test_batch_dispatch_enumerated(self):
        fx = load_enumerated_fixtures()[0]
        paths = generate_from_fixture(fx, seed=42)
        batch = ScenarioReasoningBatch(dry_run=True)
        report = batch.run_paths(paths)
        assert report["schema"] == "scenario_reasoning_batch_v1"
        assert report["path_count"] == len(paths)
        assert report["provider_mode"] == "mock"
        assert len(report["results"]) == len(paths)

    def test_mock_traces_present(self):
        fx = next(f for f in load_enumerated_fixtures() if f.fixture_id == "causal_good_bad_worst")
        paths = generate_from_fixture(fx)
        result = run_reasoning_for_path(paths[0])
        assert len(result.spans) >= 3
        assert result.reasoning_text
        assert "goal_scores" in result.to_dict() or result.goal_scores

    def test_theta_constraints_in_reasoning_loop(self):
        fx = next(f for f in load_enumerated_fixtures() if f.fixture_id == "enterprise_five_stage_ladder")
        paths = generate_from_fixture(fx)
        batch = ScenarioReasoningBatch(dry_run=True)
        report = batch.run_paths(paths)
        for row in report["results"]:
            assert row["theta"]["filing_id"] == "acme_corp_10k"
            assert row["scenario_goal"] == fx.goal
            assert row["goal_scores"]["on_target_composite"] > 0.0

    def test_exploratory_batch(self):
        fx = load_exploratory_fixtures()[0]
        paths = generate_from_exploratory_fixture(fx)
        batch = ScenarioReasoningBatch(dry_run=True)
        report = batch.run_paths(paths)
        assert report["path_count"] >= fx.min_path_count

    def test_off_target_fixture_scores_lower(self):
        batch = ScenarioReasoningBatch(dry_run=True)
        on_report = batch.run_enumerated_fixture("enterprise_five_stage_ladder")
        off_paths = [
            {
                "path_id": "off_0",
                "stage": "bad",
                "scenario_goal": on_report["results"][0]["scenario_goal"],
                "theta": on_report["results"][0]["theta"],
                "causal_slice": {},
                "scenario_type": "enterprise",
                "metadata": {},
            }
        ]
        off_paths[0]["scenario_goal"] = on_report["results"][0]["scenario_goal"]
        off_paths[0]["reasoning_text"] = "biotech S-1 minor office delay unrelated filing"
        off_result = run_reasoning_for_path(off_paths[0])
        on_composite = on_report["results"][0]["goal_scores"]["on_target_composite"]
        assert off_result.goal_scores["on_target_composite"] < on_composite

    def test_live_blocked_without_gate(self, monkeypatch):
        monkeypatch.delenv("ALLOW_LIVE_PROVIDER", raising=False)
        batch = ScenarioReasoningBatch(dry_run=False, live=True)
        with pytest.raises(RuntimeError, match="ALLOW_LIVE_PROVIDER"):
            batch.run_paths([])

    def test_run_all_fixtures(self):
        batch = ScenarioReasoningBatch(dry_run=True)
        report = batch.run_all_fixtures()
        assert report["batch_count"] >= 6
        assert len(report["enumerated_batches"]) >= 3
        assert len(report["exploratory_batches"]) >= 3
