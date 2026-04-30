"""
Composite reward composer.

Combines rule-based task reward with auxiliary monitoring signals from the CoT,
ToT, and Aha monitors into a single scalar reward used by the RLHF trainer:

    R_total = clip(R_task + α·R_cot + β·R_tot + γ·R_aha, 0, 1)

The monitoring-based bonuses encourage the model to generate structured,
expressive reasoning traces without requiring human annotation of trace quality.

Component meanings:
    R_task — causal correctness, completeness, and answer accuracy
    R_cot  — presence and depth of chain-of-thought step structure
    R_tot  — tree-of-thought branching and explicit path selection
    R_aha  — "aha moment" breakthrough signals in the trace
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional

from src.monitoring.aha_monitor import AhaMonitor
from src.monitoring.cot_monitor import CoTMonitor
from src.monitoring.tot_monitor import ToTMonitor
from src.scenarios.causal.taxonomy import CausalTheta
from src.training.causal_reward import CausalRewardFunction

logger = logging.getLogger(__name__)


class RewardComposer:
    """
    Combines task reward and monitoring-derived auxiliary rewards.

    The composite reward is clipped to ``[0, 1]`` after summation to keep it
    compatible with standard RLHF scaling.

    Args:
        task_reward_fn: An instance of :class:`CausalRewardFunction` (or any
                        callable accepting ``(prompt, trace, answer,
                        expected_answer, theta)`` and returning a float).
        alpha: Weight for CoT quality bonus ``R_cot``.  Default ``0.15``.
        beta:  Weight for ToT quality bonus ``R_tot``.  Default ``0.10``.
        gamma: Weight for Aha-moment frequency bonus ``R_aha``.  Default ``0.05``.

    Example::

        from src.training.causal_reward import CausalRewardFunction
        from src.training.reward_composer import RewardComposer

        composer = RewardComposer(CausalRewardFunction(), alpha=0.15, beta=0.10, gamma=0.05)
        result = composer.score(
            prompt="...",
            trace="Step 1: X causes Y.\\nTherefore, Y.",
            answer="The final outcome is: Y.",
            expected_answer="The final outcome is: Y.",
            theta=theta,
        )
        print(result["R_total"])   # → float in [0, 1]
        print(result["R_cot"])     # → CoT quality bonus
    """

    def __init__(
        self,
        task_reward_fn: CausalRewardFunction,
        alpha: float = 0.15,
        beta: float = 0.10,
        gamma: float = 0.05,
    ) -> None:
        self.task_reward_fn = task_reward_fn
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self._cot_monitor = CoTMonitor()
        self._tot_monitor = ToTMonitor()
        self._aha_monitor = AhaMonitor()

    def score(
        self,
        prompt: str,
        trace: str,
        answer: str,
        expected_answer: Optional[str] = None,
        theta: Optional[CausalTheta] = None,
        sample_id: Optional[Any] = None,
    ) -> Dict[str, float]:
        """
        Compute all reward components and the composite total for one sample.

        Args:
            prompt: The scenario prompt fed to the model.
            trace: The model's reasoning trace (CoT/ToT text).
            answer: The model's final answer.
            expected_answer: Ground-truth answer (optional).
            theta: :class:`CausalTheta` for domain-specific scoring (optional).
            sample_id: Optional identifier forwarded to monitors for logging.

        Returns:
            Dict with keys ``"R_task"``, ``"R_cot"``, ``"R_tot"``,
            ``"R_aha"``, ``"R_total"`` — all floats in ``[0, 1]``.
        """
        r_task = self.task_reward_fn.score(prompt, trace, answer, expected_answer, theta)
        r_cot = self._cot_reward(trace, sample_id)
        r_tot = self._tot_reward(trace, sample_id)
        r_aha = self._aha_reward(trace, sample_id)

        r_total = r_task + self.alpha * r_cot + self.beta * r_tot + self.gamma * r_aha
        r_total = min(max(r_total, 0.0), 1.0)

        breakdown = {
            "R_task": round(r_task, 4),
            "R_cot": round(r_cot, 4),
            "R_tot": round(r_tot, 4),
            "R_aha": round(r_aha, 4),
            "R_total": round(r_total, 4),
        }
        logger.debug("Reward breakdown (sample=%s): %s", sample_id, breakdown)
        return breakdown

    def score_batch(
        self,
        prompts: List[str],
        traces: List[str],
        answers: List[str],
        expected_answers: Optional[List[Optional[str]]] = None,
        thetas: Optional[List[Optional[CausalTheta]]] = None,
        sample_ids: Optional[List[Any]] = None,
    ) -> List[Dict[str, float]]:
        """
        Score a batch of ``(prompt, trace, answer)`` triples.

        Args:
            prompts: List of scenario prompts.
            traces: List of model reasoning traces.
            answers: List of model final answers.
            expected_answers: Optional per-sample ground-truth answers.
            thetas: Optional per-sample CausalTheta objects.
            sample_ids: Optional per-sample identifiers for logging.

        Returns:
            List of reward dicts (same structure as :meth:`score`).
        """
        n = len(prompts)
        expected_answers = expected_answers or [None] * n
        thetas = thetas or [None] * n
        sample_ids = sample_ids or list(range(n))

        return [
            self.score(p, t, a, ea, th, sid)
            for p, t, a, ea, th, sid in zip(
                prompts, traces, answers, expected_answers, thetas, sample_ids
            )
        ]

    # ── Monitoring reward helpers ─────────────────────────────────────────────

    def _cot_reward(self, trace: str, sample_id: Any) -> float:
        """
        Map CoT monitor output to a ``[0, 1]`` bonus.

        Reward is proportional to step count (saturates at 6 steps = 1.0).
        Returns 0.0 when no CoT structure is detected.
        """
        cot_trace = self._cot_monitor.extract(trace, sample_id=sample_id)
        if not cot_trace.has_cot:
            return 0.0
        step_score = min(cot_trace.step_count / 6.0, 1.0)
        return round(step_score, 4)

    def _tot_reward(self, trace: str, sample_id: Any) -> float:
        """
        Map ToT monitor output to a ``[0, 1]`` bonus.

        Rewards branching (60 %) and explicit path selection (40 %).
        Returns 0.0 when no tree-of-thought structure is detected.
        """
        tot_trace = self._tot_monitor.extract(trace, sample_id=sample_id)
        if not tot_trace.has_tot:
            return 0.0
        has_selection = any(n.is_selected for n in tot_trace.nodes)
        branch_score = min(tot_trace.branch_count / 3.0, 1.0)
        return round(0.6 * branch_score + 0.4 * float(has_selection), 4)

    def _aha_reward(self, trace: str, sample_id: Any) -> float:
        """
        Map Aha monitor output to a ``[0, 1]`` bonus.

        1 aha moment → 0.5; 2+ moments → 1.0.
        Returns 0.0 when no aha moments are detected.
        """
        aha_trace = self._aha_monitor.extract(trace, sample_id=sample_id)
        if not aha_trace.has_aha:
            return 0.0
        return min(aha_trace.moment_count / 2.0, 1.0)
