"""
Metrics module for Scenario Reasoner LM.

Base metric class hierarchy and registry for evaluating reasoning quality.
"""

from src.metrics.base_metrics import BaseMetric, MetricRegistry
from src.metrics.goal_preservation_metrics import (
    GoalPreservationScore,
    OnTargetReasoningMetric,
    register_goal_preservation_metrics,
)
from src.metrics.cross_scenario_coherence_metrics import (
    CrossScenarioCoherenceScore,
    SiblingNonContradictionMetric,
    register_cross_scenario_coherence_metrics,
)

__all__ = [
    "BaseMetric",
    "MetricRegistry",
    "GoalPreservationScore",
    "OnTargetReasoningMetric",
    "register_goal_preservation_metrics",
    "CrossScenarioCoherenceScore",
    "SiblingNonContradictionMetric",
    "register_cross_scenario_coherence_metrics",
]
