"""
Unit tests for goal-preservation metrics (S5-03).
"""

from __future__ import annotations

import pytest

from src.metrics.base_metrics import MetricRegistry
from src.metrics.goal_preservation_metrics import (
    GoalPreservationScore,
    OnTargetReasoningMetric,
    evaluate_fixtures,
    load_goal_preservation_fixtures,
    register_goal_preservation_metrics,
    score_goal_alignment,
)


class TestGoalPreservationFixtures:
    def test_load_fixtures(self):
        rows = load_goal_preservation_fixtures()
        assert len(rows) >= 5
        off_target = [r for r in rows if r.is_off_target_fixture]
        assert len(off_target) >= 2


class TestScoreGoalAlignment:
    def test_on_target_enterprise_scores_higher(self):
        on = score_goal_alignment(
            "Risk Factors Taiwan supply chain catastrophic scenarios from 10-K",
            "Five catastrophic enterprise scenarios grounded in acme 10-K Risk Factors",
            {"filing_id": "acme_corp_10k", "severity_floor": "high"},
        )
        off = score_goal_alignment(
            "biotech S-1 clinical trials minor office delay",
            "Five catastrophic enterprise scenarios grounded in acme 10-K Risk Factors",
            {"filing_id": "acme_corp_10k", "severity_floor": "high"},
        )
        assert on["on_target_composite"] > off["on_target_composite"]

    def test_off_target_severity_violation(self):
        scores = score_goal_alignment(
            "minor unlikely low severity",
            "Five catastrophic enterprise scenarios",
            {"severity_floor": "catastrophic"},
        )
        assert scores["theta_constraint_match"] < 0.5


class TestGoalPreservationMetrics:
    def test_registry_integration(self):
        reg = MetricRegistry()
        register_goal_preservation_metrics(reg)
        assert "goal_preservation" in reg
        assert "on_target_reasoning" in reg

    def test_evaluate_fixtures_aggregate(self):
        report = evaluate_fixtures()
        assert "aggregate" in report
        assert "goal_preservation" in report["aggregate"]
        assert report["fixture_count"] >= 5

    def test_off_target_regression_detected(self):
        report = evaluate_fixtures()
        assert report["off_target_false_positive_rate"] < 1.0

    def test_metric_update_compute(self):
        m = GoalPreservationScore()
        m.update(
            ["Risk Factors catastrophic Taiwan supply 10-K"],
            ["goal"],
            scenario_goals=["Five catastrophic enterprise scenarios grounded in acme 10-K"],
            theta_constraints=[{"filing_id": "acme_corp_10k"}],
        )
        assert m.compute() > 0.0

    def test_on_target_metric(self):
        m = OnTargetReasoningMetric(threshold=0.4)
        m.update(
            ["Step 1: A. Step 2: B. Therefore outcome."],
            ["goal"],
            scenario_goals=["Answer direct-chain causal question"],
            theta_constraints=[{"intervention_type": "direct"}],
        )
        assert m.compute() >= 0.0
