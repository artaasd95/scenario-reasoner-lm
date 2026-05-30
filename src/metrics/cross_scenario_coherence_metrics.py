"""
Cross-scenario reasoning coherence metrics (S6-04 / S7-04).

Scores whether related paths (same θ family, different severity/stage) produce
coherent causal claims and non-contradictory conclusions.
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
DEFAULT_FIXTURES_PATH = _REPO_ROOT / "data" / "eval" / "cross_scenario_coherence_fixtures.jsonl"

_CONTRADICTION_PAIRS = [
    (r"\bcatastrophic\b", r"\bminor|unlikely|low severity\b"),
    (r"\bincrease|grow|expand\b", r"\bdecrease|shrink|contract\b"),
    (r"\bwithout intervention\b", r"\bdirect chain always\b"),
    (r"\bsupply chain\b", r"\bbiot(ech)?\s+s-1\b"),
]

_SEVERITY_ORDER = {
    "good": 0,
    "baseline": 0,
    "neutral": 1,
    "elevated": 1,
    "bad": 2,
    "severe": 2,
    "critical": 3,
    "worst": 4,
    "catastrophic": 4,
}


@dataclass
class SiblingPathRecord:
    fixture_id: str
    theta_family_id: str
    scenario_goal: str
    paths: List[Dict[str, Any]]
    tags: List[str] = field(default_factory=list)
    expected_coherent: bool = True

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "SiblingPathRecord":
        return cls(
            fixture_id=data["fixture_id"],
            theta_family_id=data["theta_family_id"],
            scenario_goal=data["scenario_goal"],
            paths=list(data["paths"]),
            tags=list(data.get("tags", [])),
            expected_coherent=bool(data.get("expected_coherent", True)),
        )


def load_coherence_fixtures(path: Path | str | None = None) -> List[SiblingPathRecord]:
    fixture_path = Path(path) if path else DEFAULT_FIXTURES_PATH
    if not fixture_path.is_file():
        raise FileNotFoundError(f"Cross-scenario coherence fixtures not found: {fixture_path}")
    records: List[SiblingPathRecord] = []
    with open(fixture_path, encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                records.append(SiblingPathRecord.from_dict(json.loads(line)))
    return records


def _extract_conclusion(text: str) -> str:
    m = re.search(r"(therefore|thus|hence|in conclusion)[,:]?\s*(.+)", text, re.I)
    return m.group(2).strip().lower() if m else text.lower()[-120:]


def _pairwise_contradiction(a: str, b: str) -> float:
    """Return 1.0 when a contradiction heuristic fires between two texts."""
    for pos, neg in _CONTRADICTION_PAIRS:
        a_pos = bool(re.search(pos, a, re.I))
        a_neg = bool(re.search(neg, a, re.I))
        b_pos = bool(re.search(pos, b, re.I))
        b_neg = bool(re.search(neg, b, re.I))
        if (a_pos and b_neg) or (a_neg and b_pos):
            return 1.0
    return 0.0


def _severity_monotonic(paths: List[Dict[str, Any]]) -> float:
    """Score whether severity ranks increase along sibling stages."""
    ranks = []
    for p in paths:
        stage = p.get("stage", "").lower()
        meta_rank = p.get("metadata", {}).get("severity_rank")
        if meta_rank is not None:
            ranks.append(int(meta_rank))
        elif stage in _SEVERITY_ORDER:
            ranks.append(_SEVERITY_ORDER[stage])
    if len(ranks) < 2:
        return 1.0
    violations = sum(1 for i in range(len(ranks) - 1) if ranks[i] > ranks[i + 1])
    return max(0.0, 1.0 - violations / max(len(ranks) - 1, 1))


def score_sibling_coherence(paths: Sequence[Dict[str, Any]]) -> Dict[str, float]:
    """
    Heuristic coherence across sibling paths without live provider calls.

    Returns:
        ``claim_overlap``, ``non_contradiction``, ``severity_ordering``, ``coherence_composite``
    """
    texts = [p.get("reasoning_text", p.get("conclusion", "")) for p in paths]
    if not texts:
        return {
            "claim_overlap": 0.0,
            "non_contradiction": 0.0,
            "severity_ordering": 0.0,
            "coherence_composite": 0.0,
        }

    conclusions = [_extract_conclusion(t) for t in texts]
    tokens_sets = [
        {w for w in re.findall(r"[a-z]{4,}", c) if len(w) > 4} for c in conclusions
    ]
    overlaps: List[float] = []
    contradictions: List[float] = []
    for i in range(len(texts)):
        for j in range(i + 1, len(texts)):
            ti, tj = tokens_sets[i], tokens_sets[j]
            if ti or tj:
                overlaps.append(len(ti & tj) / max(len(ti | tj), 1))
            contradictions.append(_pairwise_contradiction(texts[i], texts[j]))

    claim_overlap = round(mean(overlaps) if overlaps else 0.5, 4)
    non_contradiction = round(1.0 - mean(contradictions) if contradictions else 1.0, 4)
    severity_ordering = round(_severity_monotonic(list(paths)), 4)
    composite = round(
        0.35 * claim_overlap + 0.40 * non_contradiction + 0.25 * severity_ordering,
        4,
    )
    return {
        "claim_overlap": claim_overlap,
        "non_contradiction": non_contradiction,
        "severity_ordering": severity_ordering,
        "coherence_composite": composite,
    }


class CrossScenarioCoherenceScore(BaseMetric):
    """Mean coherence composite across sibling path sets. Name: ``cross_scenario_coherence``."""

    name: str = "cross_scenario_coherence"

    def __init__(self) -> None:
        super().__init__()
        self._scores: List[float] = []

    def update(
        self,
        predictions: List[str],
        references: List[str],
        *,
        sibling_paths: Optional[List[List[Dict[str, Any]]]] = None,
        **kwargs: Any,
    ) -> None:
        if sibling_paths:
            for paths in sibling_paths:
                self._scores.append(score_sibling_coherence(paths)["coherence_composite"])
        elif predictions:
            paths = [{"reasoning_text": p} for p in predictions]
            self._scores.append(score_sibling_coherence(paths)["coherence_composite"])

    def compute(self) -> float:
        if not self._scores:
            return 0.0
        return round(mean(self._scores), 4)

    def reset(self) -> None:
        self._scores = []


class SiblingNonContradictionMetric(BaseMetric):
    """Binary-style non-contradiction rate across siblings. Name: ``sibling_non_contradiction``."""

    name: str = "sibling_non_contradiction"
    threshold: float = 0.7

    def __init__(self, threshold: float = 0.7) -> None:
        super().__init__()
        self.threshold = threshold
        self._hits = 0
        self._total = 0

    def update(
        self,
        predictions: List[str],
        references: List[str],
        *,
        sibling_paths: Optional[List[List[Dict[str, Any]]]] = None,
        **kwargs: Any,
    ) -> None:
        groups = sibling_paths or [[{"reasoning_text": p} for p in predictions]]
        for paths in groups:
            score = score_sibling_coherence(paths)["non_contradiction"]
            self._hits += int(score >= self.threshold)
            self._total += 1

    def compute(self) -> float:
        if not self._total:
            return 0.0
        return round(self._hits / self._total, 4)

    def reset(self) -> None:
        self._hits = 0
        self._total = 0


def evaluate_fixtures(
    records: Optional[Sequence[SiblingPathRecord]] = None,
    *,
    coherence_threshold: float = 0.55,
) -> Dict[str, Any]:
    """Score bundled sibling fixtures for regression without live providers."""
    rows = list(records) if records is not None else load_coherence_fixtures()
    per_fixture: List[Dict[str, Any]] = []
    cs = CrossScenarioCoherenceScore()
    nc = SiblingNonContradictionMetric(threshold=0.7)

    for rec in rows:
        scores = score_sibling_coherence(rec.paths)
        cs.update([], [], sibling_paths=[rec.paths])
        nc.update([], [], sibling_paths=[rec.paths])
        coherent = scores["coherence_composite"] >= coherence_threshold
        per_fixture.append(
            {
                "fixture_id": rec.fixture_id,
                "theta_family_id": rec.theta_family_id,
                "tags": rec.tags,
                "scores": scores,
                "expected_coherent": rec.expected_coherent,
                "passes": coherent if rec.expected_coherent else not coherent,
            }
        )

    contradictory = [r for r in per_fixture if not r["expected_coherent"]]
    false_positive_rate = (
        sum(1 for r in contradictory if r["scores"]["coherence_composite"] >= coherence_threshold)
        / len(contradictory)
        if contradictory
        else 0.0
    )

    return {
        "aggregate": {
            "cross_scenario_coherence": cs.compute(),
            "sibling_non_contradiction": nc.compute(),
        },
        "per_fixture": per_fixture,
        "contradictory_false_positive_rate": round(false_positive_rate, 4),
        "fixture_count": len(per_fixture),
    }


def register_cross_scenario_coherence_metrics(registry: Any) -> None:
    """Register S6/S7 coherence metrics on an existing MetricRegistry."""
    registry.register(CrossScenarioCoherenceScore())
    registry.register(SiblingNonContradictionMetric())
