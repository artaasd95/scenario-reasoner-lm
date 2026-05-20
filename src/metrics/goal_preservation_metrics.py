"""
Goal-preservation and on-target reasoning metrics (S5-03).

Scores whether reasoning stays aligned with scenario goals and θ constraints.
Integrates with :class:`~src.metrics.base_metrics.MetricRegistry` alongside
causal metrics; complements enterprise rubric in ``eval_metrics``.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from pathlib import Path
from statistics import mean
from typing import Any, Dict, List, Optional, Sequence

from src.metrics.base_metrics import BaseMetric

_REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_FIXTURES_PATH = _REPO_ROOT / "data" / "eval" / "goal_preservation_fixtures.jsonl"


@dataclass
class GoalPreservationRecord:
    fixture_id: str
    scenario_goal: str
    theta_constraints: Dict[str, Any]
    reasoning_text: str
    tags: List[str] = field(default_factory=list)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "GoalPreservationRecord":
        return cls(
            fixture_id=data["fixture_id"],
            scenario_goal=data["scenario_goal"],
            theta_constraints=dict(data.get("theta_constraints", {})),
            reasoning_text=data["reasoning_text"],
            tags=list(data.get("tags", [])),
        )

    @property
    def is_off_target_fixture(self) -> bool:
        return "off_target" in self.tags


def load_goal_preservation_fixtures(path: Path | str | None = None) -> List[GoalPreservationRecord]:
    fixture_path = Path(path) if path else DEFAULT_FIXTURES_PATH
    if not fixture_path.is_file():
        raise FileNotFoundError(f"Goal preservation fixtures not found: {fixture_path}")
    records: List[GoalPreservationRecord] = []
    with open(fixture_path, encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                records.append(GoalPreservationRecord.from_dict(json.loads(line)))
    return records


def score_goal_alignment(
    reasoning_text: str,
    scenario_goal: str,
    theta_constraints: Optional[Dict[str, Any]] = None,
) -> Dict[str, float]:
    """
    Heuristic alignment score in [0, 1] without live provider calls.

    Returns:
        ``goal_overlap``, ``theta_constraint_match``, ``on_target_composite``
    """
    text = reasoning_text.lower()
    goal_tokens = {t for t in re.findall(r"[a-z]{4,}", scenario_goal.lower()) if len(t) > 4}
    overlap = 0.0
    if goal_tokens:
        hits = sum(1 for t in goal_tokens if t in text)
        overlap = min(hits / max(len(goal_tokens) * 0.25, 1), 1.0)

    theta_match = 1.0
    constraints = theta_constraints or {}
    if "filing_id" in constraints and "acme" in constraints["filing_id"]:
        if "biotech" in text or "s-1" in text:
            theta_match *= 0.2
        elif "10-k" not in text and "risk factors" not in text and "supply" not in text:
            theta_match *= 0.6
    if constraints.get("severity_floor") == "catastrophic":
        if re.search(r"\b(low severity|minor|unlikely)\b", text):
            theta_match *= 0.2
    if constraints.get("intervention_type") == "counterfactual":
        if not re.search(r"without|if not|had not|counterfactual", text, re.I):
            theta_match *= 0.3
    if constraints.get("intervention_type") == "direct":
        if re.search(r"without|if not|counterfactual", text, re.I):
            theta_match *= 0.5

    composite = round(0.55 * overlap + 0.45 * theta_match, 4)
    return {
        "goal_overlap": round(overlap, 4),
        "theta_constraint_match": round(theta_match, 4),
        "on_target_composite": composite,
    }


class GoalPreservationScore(BaseMetric):
    """Mean on-target composite across accumulated samples. Name: ``goal_preservation``."""

    name: str = "goal_preservation"

    def __init__(self) -> None:
        super().__init__()
        self._scores: List[float] = []

    def update(
        self,
        predictions: List[str],
        references: List[str],
        *,
        scenario_goals: Optional[List[str]] = None,
        theta_constraints: Optional[List[Dict[str, Any]]] = None,
        **kwargs: Any,
    ) -> None:
        goals = scenario_goals or references
        thetas = theta_constraints or [{}] * len(predictions)
        for pred, goal, theta in zip(predictions, goals, thetas):
            result = score_goal_alignment(pred, goal, theta)
            self._scores.append(result["on_target_composite"])

    def compute(self) -> float:
        if not self._scores:
            return 0.0
        return round(mean(self._scores), 4)

    def reset(self) -> None:
        self._scores = []


class OnTargetReasoningMetric(BaseMetric):
    """
    Binary-style on-target rate: composite >= threshold.

    Name: ``on_target_reasoning``
    """

    name: str = "on_target_reasoning"
    threshold: float = 0.55

    def __init__(self, threshold: float = 0.55) -> None:
        super().__init__()
        self.threshold = threshold
        self._hits = 0
        self._total = 0

    def update(
        self,
        predictions: List[str],
        references: List[str],
        *,
        scenario_goals: Optional[List[str]] = None,
        theta_constraints: Optional[List[Dict[str, Any]]] = None,
        **kwargs: Any,
    ) -> None:
        goals = scenario_goals or references
        thetas = theta_constraints or [{}] * len(predictions)
        for pred, goal, theta in zip(predictions, goals, thetas):
            composite = score_goal_alignment(pred, goal, theta)["on_target_composite"]
            self._hits += int(composite >= self.threshold)
            self._total += 1

    def compute(self) -> float:
        if not self._total:
            return 0.0
        return round(self._hits / self._total, 4)

    def reset(self) -> None:
        self._hits = 0
        self._total = 0


def evaluate_fixtures(
    records: Optional[Sequence[GoalPreservationRecord]] = None,
    *,
    on_target_threshold: float = 0.55,
) -> Dict[str, Any]:
    """Score bundled fixtures; useful for regression without live providers."""
    rows = list(records) if records is not None else load_goal_preservation_fixtures()
    registry_scores: Dict[str, float] = {}
    per_fixture: List[Dict[str, Any]] = []

    gp = GoalPreservationScore()
    ot = OnTargetReasoningMetric(threshold=on_target_threshold)
    for rec in rows:
        scores = score_goal_alignment(
            rec.reasoning_text, rec.scenario_goal, rec.theta_constraints
        )
        gp.update([rec.reasoning_text], [rec.scenario_goal], scenario_goals=[rec.scenario_goal], theta_constraints=[rec.theta_constraints])
        ot.update([rec.reasoning_text], [rec.scenario_goal], scenario_goals=[rec.scenario_goal], theta_constraints=[rec.theta_constraints])
        per_fixture.append(
            {
                "fixture_id": rec.fixture_id,
                "tags": rec.tags,
                "scores": scores,
                "expected_on_target": "off_target" not in rec.tags,
            }
        )

    registry_scores["goal_preservation"] = gp.compute()
    registry_scores["on_target_reasoning"] = ot.compute()

    off_target = [r for r in per_fixture if not r["expected_on_target"]]
    off_target_fail_rate = (
        sum(1 for r in off_target if r["scores"]["on_target_composite"] >= on_target_threshold)
        / len(off_target)
        if off_target
        else 0.0
    )

    return {
        "aggregate": registry_scores,
        "per_fixture": per_fixture,
        "off_target_false_positive_rate": round(off_target_fail_rate, 4),
        "fixture_count": len(per_fixture),
    }


def register_goal_preservation_metrics(registry: Any) -> None:
    """Register S5 goal metrics on an existing :class:`~src.metrics.base_metrics.MetricRegistry`."""
    registry.register(GoalPreservationScore())
    registry.register(OnTargetReasoningMetric())
