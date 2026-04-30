"""
Causal-domain evaluation metrics.

Three :class:`~src.metrics.base_metrics.BaseMetric` subclasses that assess
trajectory quality beyond final-answer correctness:

    1. :class:`CausalChainAccuracy`      — logical consistency of causal steps
    2. :class:`CounterfactualValidityScore` — correct do-calculus intervention handling
    3. :class:`TrajectoryConsistency`    — absence of contradictions across steps

All three follow the accumulate-then-compute pattern from :class:`BaseMetric`
and register cleanly with :class:`~src.metrics.base_metrics.MetricRegistry`.
"""

from __future__ import annotations

import re
from statistics import mean
from typing import Any, Dict, List, Optional

from src.metrics.base_metrics import BaseMetric


class CausalChainAccuracy(BaseMetric):
    """
    Measures whether reasoning traces contain logically ordered causal steps.

    Scoring per sample:
        * Each detected ``"Step N: ..."`` line earns partial credit.
        * A final conclusion marker (``"Therefore"``, ``"Thus"``, etc.) earns a bonus.
        * Circular patterns (``"X causes X"``) reduce the score.

    Name: ``"causal_chain_accuracy"``

    Example::

        metric = CausalChainAccuracy()
        metric.update(predictions=["Step 1: ...\\nTherefore, Z."], references=["Z."])
        result = metric.compute()   # → float in [0, 1]
    """

    name: str = "causal_chain_accuracy"

    _STEP_RE = re.compile(r"step\s*\d+", re.IGNORECASE)
    _CONCLUSION_RE = re.compile(
        r"therefore|thus|hence|in conclusion|the final outcome", re.IGNORECASE
    )
    _CIRCULAR_RE = re.compile(
        r"(\b\w[\w\s]{2,30}?\b)\s+(?:causes|leads to|results in)\s+\1",
        re.IGNORECASE,
    )

    def __init__(self) -> None:
        super().__init__()
        self._scores: List[float] = []

    def update(
        self,
        predictions: List[str],
        references: List[str],
        **kwargs: Any,
    ) -> None:
        for pred in predictions:
            self._scores.append(self._score_trace(pred))

    def compute(self) -> float:
        if not self._scores:
            return 0.0
        return round(mean(self._scores), 4)

    def reset(self) -> None:
        self._scores = []

    # ── Internals ─────────────────────────────────────────────────────────────

    def _score_trace(self, trace: str) -> float:
        steps = len(self._STEP_RE.findall(trace))
        has_conclusion = bool(self._CONCLUSION_RE.search(trace))
        has_cycle = bool(self._CIRCULAR_RE.search(trace))

        step_score = min(steps / 3.0, 1.0)
        score = 0.7 * step_score + 0.3 * float(has_conclusion)
        if has_cycle:
            score *= 0.3
        return round(score, 4)


class CounterfactualValidityScore(BaseMetric):
    """
    Measures whether counterfactual traces correctly address the intervention.

    Applicable only to ``"counterfactual"`` and ``"confounded"`` scenarios.
    For direct-chain predictions, the score is always ``1.0`` (not applicable).

    Scoring per sample:
        * Addresses the intervention (``"without"``, ``"if not"``, etc.): +0.5
        * Provides a clear yes/no answer polarity: +0.5

    Name: ``"counterfactual_validity"``

    Example::

        metric = CounterfactualValidityScore()
        metric.update(
            predictions=["No. Without X, Y would not occur."],
            references=["No."],
            intervention_types=["counterfactual"],
        )
        result = metric.compute()  # → float in [0, 1]
    """

    name: str = "counterfactual_validity"

    _INTERVENTION_RE = re.compile(
        r"without|if not|had not|absent|removed|eliminated", re.IGNORECASE
    )
    _YES_RE = re.compile(r"\byes\b", re.IGNORECASE)
    _NO_RE = re.compile(r"\bno\b|would not|wouldn't", re.IGNORECASE)

    def __init__(self) -> None:
        super().__init__()
        self._scores: List[float] = []

    def update(
        self,
        predictions: List[str],
        references: List[str],
        intervention_types: Optional[List[str]] = None,
        **kwargs: Any,
    ) -> None:
        types = intervention_types or ["direct"] * len(predictions)
        for pred, itype in zip(predictions, types):
            if itype == "direct":
                self._scores.append(1.0)
            else:
                self._scores.append(self._score_cf(pred))

    def compute(self) -> float:
        if not self._scores:
            return 0.0
        return round(mean(self._scores), 4)

    def reset(self) -> None:
        self._scores = []

    def _score_cf(self, pred: str) -> float:
        addresses = float(bool(self._INTERVENTION_RE.search(pred)))
        clear_polarity = float(
            bool(self._YES_RE.search(pred)) or bool(self._NO_RE.search(pred))
        )
        return 0.5 * addresses + 0.5 * clear_polarity


class TrajectoryConsistency(BaseMetric):
    """
    Detects contradictions and logical inconsistencies across reasoning steps.

    A trajectory is penalized when it contains:
        * Explicit contrastive contradiction (``"However, … Therefore"``).
        * Circular causation (``"X causes X"``).
        * Inconsistent polarity between trace and answer
          (trace says "yes" but answer says "no").

    Score: ``1.0`` = fully consistent, lower = inconsistent.

    Name: ``"trajectory_consistency"``

    Example::

        metric = TrajectoryConsistency()
        metric.update(
            predictions=["Step 1: X causes Y.\\nTherefore, Y."],
            references=["Y."],
        )
        result = metric.compute()  # → float in [0, 1]
    """

    name: str = "trajectory_consistency"

    _CONTRADICTION_RE = re.compile(
        r"(?:however|but|yet|contradicts|despite this)"
        r".{0,80}"
        r"(?:therefore|thus|the answer is)",
        re.IGNORECASE | re.DOTALL,
    )
    _CIRCULAR_RE = re.compile(
        r"(\b\w[\w\s]{2,30}?\b)\s+(?:causes|leads to|results in)\s+\1",
        re.IGNORECASE,
    )
    _YES_RE = re.compile(r"\byes\b", re.IGNORECASE)
    _NO_RE = re.compile(r"\bno\b|would not|wouldn't", re.IGNORECASE)

    def __init__(self) -> None:
        super().__init__()
        self._scores: List[float] = []

    def update(
        self,
        predictions: List[str],
        references: List[str],
        **kwargs: Any,
    ) -> None:
        for pred, ref in zip(predictions, references):
            self._scores.append(self._score_consistency(pred, ref))

    def compute(self) -> float:
        if not self._scores:
            return 0.0
        return round(mean(self._scores), 4)

    def reset(self) -> None:
        self._scores = []

    def _score_consistency(self, pred: str, ref: str) -> float:
        has_contradiction = bool(self._CONTRADICTION_RE.search(pred))
        has_cycle = bool(self._CIRCULAR_RE.search(pred))

        # Polarity flip: trace says yes but answer says no (or vice versa)
        trace_yes = bool(self._YES_RE.search(pred))
        trace_no = bool(self._NO_RE.search(pred))
        ref_yes = bool(self._YES_RE.search(ref))
        ref_no = bool(self._NO_RE.search(ref))
        polarity_flip = (trace_yes and ref_no) or (trace_no and ref_yes)

        penalty = (
            0.5 * float(has_contradiction)
            + 0.3 * float(has_cycle)
            + 0.2 * float(polarity_flip)
        )
        return round(max(0.0, 1.0 - penalty), 4)
