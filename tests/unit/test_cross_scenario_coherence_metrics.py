"""
Unit tests for cross-scenario coherence metrics (S6-04 / S7-04).
"""

from __future__ import annotations

from src.metrics.base_metrics import MetricRegistry
from src.metrics.cross_scenario_coherence_metrics import (
    CrossScenarioCoherenceScore,
    SiblingNonContradictionMetric,
    evaluate_fixtures,
    load_coherence_fixtures,
    register_cross_scenario_coherence_metrics,
    score_sibling_coherence,
)


class TestCoherenceFixtures:
    def test_load_fixtures(self):
        rows = load_coherence_fixtures()
        assert len(rows) >= 5
        contradictory = [r for r in rows if not r.expected_coherent]
        assert len(contradictory) >= 2


class TestScoreSiblingCoherence:
    def test_coherent_ladder_scores_higher(self):
        coherent = score_sibling_coherence(
            [
                {"stage": "good", "metadata": {"severity_rank": 0}, "reasoning_text": "Step 1: A. Therefore stable."},
                {"stage": "bad", "metadata": {"severity_rank": 2}, "reasoning_text": "Step 1: A. Therefore adverse."},
                {"stage": "worst", "metadata": {"severity_rank": 3}, "reasoning_text": "Step 1: A. Therefore catastrophic."},
            ]
        )
        contradictory = score_sibling_coherence(
            [
                {"stage": "good", "metadata": {"severity_rank": 0}, "reasoning_text": "Supply chain Risk Factors 10-K catastrophic."},
                {"stage": "worst", "metadata": {"severity_rank": 3}, "reasoning_text": "Minor low severity unlikely biotech S-1."},
            ]
        )
        assert coherent["coherence_composite"] > contradictory["coherence_composite"]
        assert contradictory["non_contradiction"] < coherent["non_contradiction"]

    def test_severity_ordering_violation(self):
        scores = score_sibling_coherence(
            [
                {"stage": "catastrophic", "metadata": {"severity_rank": 4}, "reasoning_text": "Therefore catastrophic."},
                {"stage": "good", "metadata": {"severity_rank": 0}, "reasoning_text": "Therefore minor."},
            ]
        )
        assert scores["severity_ordering"] < 1.0


class TestCoherenceMetrics:
    def test_registry_integration(self):
        reg = MetricRegistry()
        register_cross_scenario_coherence_metrics(reg)
        assert "cross_scenario_coherence" in reg
        assert "sibling_non_contradiction" in reg

    def test_evaluate_fixtures_aggregate(self):
        report = evaluate_fixtures()
        assert "aggregate" in report
        assert "cross_scenario_coherence" in report["aggregate"]
        assert report["fixture_count"] >= 5

    def test_contradictory_regression_detected(self):
        report = evaluate_fixtures()
        passes = [r["passes"] for r in report["per_fixture"]]
        assert all(passes)
        assert report["contradictory_false_positive_rate"] < 1.0

    def test_metric_update_compute(self):
        m = CrossScenarioCoherenceScore()
        paths = load_coherence_fixtures()[0].paths
        m.update([], [], sibling_paths=[paths])
        assert m.compute() > 0.0

    def test_non_contradiction_metric(self):
        m = SiblingNonContradictionMetric(threshold=0.5)
        coherent = load_coherence_fixtures()[0].paths
        m.update([], [], sibling_paths=[coherent])
        assert m.compute() >= 0.0
