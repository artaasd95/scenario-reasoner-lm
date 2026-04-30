"""
Rule-based causal task reward function.

Scores a ``(prompt, trace, answer)`` triple for causal reasoning correctness
without requiring a trained reward model, consuming no extra GPU memory.

Scoring components (weights sum to 1):
    1. **Chain completeness** (35 %) — does the trace contain the expected number
       of causal reasoning steps and a final conclusion?
    2. **Logical consistency** (25 %) — absence of circular causation or
       self-contradictory conclusion patterns.
    3. **Counterfactual validity** (20 %) — for counterfactual/confounded scenarios,
       does the trace address the do-calculus intervention and state a clear yes/no?
    4. **Answer correctness** (20 %) — key entity overlap between generated and
       expected answer; falls back to 0.5 when no expected answer is available.
"""

from __future__ import annotations

import re
from typing import Optional

from src.scenarios.causal.taxonomy import CausalTheta


class CausalRewardFunction:
    """
    Rule-based reward function for causal/counterfactual reasoning traces.

    Returns ``R_task ∈ [0, 1]`` by combining four sub-scores with configurable
    weights.  All computation is regex-based — no model inference required.

    Args:
        weight_completeness: Weight for chain completeness sub-score.
        weight_consistency:  Weight for logical consistency sub-score.
        weight_cf_validity:  Weight for counterfactual validity sub-score.
        weight_answer:       Weight for answer correctness sub-score.

    Example::

        reward_fn = CausalRewardFunction()
        score = reward_fn.score(
            prompt="...",
            trace="Step 1: X causes Y.\\nStep 2: Y causes Z.\\nTherefore, Z.",
            answer="The final outcome is: Z.",
            expected_answer="The final outcome is: Z.",
            theta=CausalTheta(chain_length=2, intervention_type="direct"),
        )
        # → float in [0, 1]
    """

    _STEP_RE = re.compile(r"step\s*\d+", re.IGNORECASE)
    _CONCLUSION_RE = re.compile(
        r"therefore|thus|hence|in conclusion|the final outcome|the answer",
        re.IGNORECASE,
    )
    _CF_ANSWER_YES_RE = re.compile(r"\byes\b", re.IGNORECASE)
    _CF_ANSWER_NO_RE = re.compile(r"\bno\b", re.IGNORECASE)
    _NOT_OCCUR_RE = re.compile(
        r"would not|wouldn't|would not have|wouldn't have", re.IGNORECASE
    )
    _CONTRADICTION_RE = re.compile(
        r"(?:however|but|yet|on the other hand|contradicts|despite this)"
        r".{0,80}"
        r"(?:therefore|thus|the answer is)",
        re.IGNORECASE | re.DOTALL,
    )
    _CIRCULAR_RE = re.compile(
        r"(\b\w[\w\s]{2,30}?\b)\s+(?:causes|leads to|results in)\s+\1",
        re.IGNORECASE,
    )
    _INTERVENTION_RE = re.compile(
        r"without|if not|had not|absent|removed|eliminated",
        re.IGNORECASE,
    )

    def __init__(
        self,
        weight_completeness: float = 0.35,
        weight_consistency: float = 0.25,
        weight_cf_validity: float = 0.20,
        weight_answer: float = 0.20,
    ) -> None:
        total = weight_completeness + weight_consistency + weight_cf_validity + weight_answer
        self.w_completeness = weight_completeness / total
        self.w_consistency = weight_consistency / total
        self.w_cf_validity = weight_cf_validity / total
        self.w_answer = weight_answer / total

    def score(
        self,
        prompt: str,
        trace: str,
        answer: str,
        expected_answer: Optional[str] = None,
        theta: Optional[CausalTheta] = None,
    ) -> float:
        """
        Compute ``R_task ∈ [0, 1]`` for a single ``(prompt, trace, answer)`` triple.

        Args:
            prompt: The scenario prompt fed to the model (used for context).
            trace: The model's reasoning trace output.
            answer: The model's final answer output.
            expected_answer: Ground-truth answer string (optional).
            theta: :class:`CausalTheta` used to generate the scenario (optional).
                   When provided, sub-scorers use it for domain-specific checks.

        Returns:
            Scalar task reward in ``[0, 1]``.
        """
        s_complete = self._chain_completeness(trace, theta)
        s_consist = self._logical_consistency(trace)
        s_cf = self._counterfactual_validity(trace, answer, theta)
        s_answer = self._answer_correctness(answer, expected_answer)

        return round(
            self.w_completeness * s_complete
            + self.w_consistency * s_consist
            + self.w_cf_validity * s_cf
            + self.w_answer * s_answer,
            4,
        )

    # ── Sub-scorers ──────────────────────────────────────────────────────────

    def _chain_completeness(
        self,
        trace: str,
        theta: Optional[CausalTheta],
    ) -> float:
        """Score how completely the trace covers required reasoning steps."""
        steps_found = len(self._STEP_RE.findall(trace))
        has_conclusion = bool(self._CONCLUSION_RE.search(trace))

        if theta is not None:
            expected_steps = theta.chain_length
            step_ratio = min(steps_found / max(expected_steps, 1), 1.0)
        else:
            step_ratio = min(steps_found / 2, 1.0)

        return round(0.7 * step_ratio + 0.3 * float(has_conclusion), 4)

    def _logical_consistency(self, trace: str) -> float:
        """
        Penalize circular causation or directly contradictory conclusions.

        Uses surface-level regex heuristics that are fast and require no
        semantic model.  Returns 1.0 (no penalty) when no issues are detected.
        """
        has_contradiction = bool(self._CONTRADICTION_RE.search(trace))
        has_cycle = bool(self._CIRCULAR_RE.search(trace))
        if has_contradiction or has_cycle:
            return 0.2
        return 1.0

    def _counterfactual_validity(
        self,
        trace: str,
        answer: str,
        theta: Optional[CausalTheta],
    ) -> float:
        """
        For counterfactual/confounded scenarios, verify the trace addresses the
        do-calculus intervention and provides a clear yes/no conclusion.

        Returns ``1.0`` for direct-chain scenarios (not applicable).
        """
        if theta is None or theta.intervention_type == "direct":
            return 1.0

        combined = (trace + " " + answer).lower()
        addresses_intervention = bool(self._INTERVENTION_RE.search(combined))
        has_clear_answer = bool(
            self._CF_ANSWER_YES_RE.search(answer)
            or self._CF_ANSWER_NO_RE.search(answer)
            or self._NOT_OCCUR_RE.search(answer)
        )
        return 0.5 * float(addresses_intervention) + 0.5 * float(has_clear_answer)

    def _answer_correctness(
        self,
        answer: str,
        expected_answer: Optional[str],
    ) -> float:
        """
        Compare generated answer against the expected answer.

        * Returns ``1.0`` on exact match (case-insensitive, stripped).
        * Returns key entity overlap ratio for partial matches.
        * Returns ``0.5`` (neutral) when no expected answer is available.
        """
        if expected_answer is None:
            return 0.5

        answer_lower = answer.strip().lower()
        expected_lower = expected_answer.strip().lower()

        if answer_lower == expected_lower:
            return 1.0

        expected_words = set(re.findall(r"\b\w{4,}\b", expected_lower))
        answer_words = set(re.findall(r"\b\w{4,}\b", answer_lower))

        if not expected_words:
            return 0.5

        overlap = len(expected_words & answer_words) / len(expected_words)
        return round(overlap, 4)
