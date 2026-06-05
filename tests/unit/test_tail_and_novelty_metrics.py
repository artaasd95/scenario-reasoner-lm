"""Tail / non-possible scenario metrics (S10-02)."""

from __future__ import annotations

from src.metrics.tail_scenario_metrics import tag_tail_scenario, tail_scenario_metrics
from src.scenarios.feasibility import assess_path_feasibility


class TestTailAndFeasibility:
    def test_severe_stress_is_tail(self):
        tag = tag_tail_scenario({"theta": {"stress_regime": "severe"}})
        assert tag["tail_tag"] == "tail"

    def test_non_possible_flag(self):
        tag = tag_tail_scenario({"non_possible": True})
        assert tag["tail_tag"] == "non_possible"

    def test_feasibility_infeasible(self):
        result = assess_path_feasibility({"infeasible": True})
        assert result["feasible"] is False

    def test_aggregate_rates(self):
        rows = [
            {"theta": {"stress_regime": "severe"}},
            {"non_possible": True},
            {"theta": {}},
        ]
        metrics = tail_scenario_metrics(rows)
        assert metrics["tail_rate"] > 0
