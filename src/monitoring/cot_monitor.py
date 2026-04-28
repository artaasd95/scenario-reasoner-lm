"""
Chain-of-Thought (CoT) monitor for Scenario Reasoner LM.

Detects, extracts, and logs chain-of-thought reasoning patterns from
LLM output text.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Pattern

logger = logging.getLogger(__name__)


@dataclass
class CoTStep:
    """A single extracted step in a chain-of-thought trace."""

    index: int
    text: str
    is_conclusion: bool = False
    confidence: Optional[float] = None


@dataclass
class CoTTrace:
    """A complete chain-of-thought trace extracted from one model output."""

    raw_text: str
    steps: List[CoTStep] = field(default_factory=list)
    has_cot: bool = False
    step_count: int = 0
    sample_id: Optional[Any] = None

    def __post_init__(self) -> None:
        self.step_count = len(self.steps)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "has_cot": self.has_cot,
            "step_count": self.step_count,
            "sample_id": self.sample_id,
            "steps": [
                {
                    "index": s.index,
                    "text": s.text,
                    "is_conclusion": s.is_conclusion,
                    "confidence": s.confidence,
                }
                for s in self.steps
            ],
        }


class CoTMonitor:
    """
    Monitors and extracts Chain-of-Thought (CoT) patterns from LLM outputs.

    Detects step-by-step reasoning traces such as:
    * ``"Step 1: ..."`` / ``"Step 2: ..."``
    * ``"First, ... Second, ... Finally, ..."``
    * ``"Therefore, ..."`` / ``"In conclusion, ..."``

    Example::

        monitor = CoTMonitor()
        trace   = monitor.extract("Step 1: ... Step 2: ... Therefore: ...")
        print(trace.step_count)

        for batch_outputs in outputs:
            monitor.update(batch_outputs)

        stats = monitor.get_stats()

    Args:
        step_patterns: List of regex patterns identifying reasoning steps.
        conclusion_patterns: Patterns that signal the conclusion of a CoT trace.
        log_every: Log aggregated statistics every *n* calls to :meth:`update`.
    """

    DEFAULT_STEP_PATTERNS: List[str] = [
        r"(?:Step\s*\d+|step\s*\d+)[:\.\)]\s*(.+?)(?=(?:Step\s*\d+|step\s*\d+)[:\.\)]|$)",
        r"(?:First|Second|Third|Fourth|Fifth|Next|Then)[,:]?\s*(.+?)(?=(?:First|Second|Third|Fourth|Fifth|Next|Then|Finally)[,:]|$)",
        r"\d+\.\s*(.+?)(?=\d+\.|$)",
    ]

    DEFAULT_CONCLUSION_PATTERNS: List[str] = [
        r"(?:Therefore|Thus|Hence|In conclusion|Finally|So)[,:]?\s*(.+)",
        r"(?:The answer is|My answer is)[:\s]*(.+)",
    ]

    def __init__(
        self,
        step_patterns: Optional[List[str]] = None,
        conclusion_patterns: Optional[List[str]] = None,
        log_every: int = 100,
    ) -> None:
        self._step_patterns: List[Pattern] = [
            re.compile(p, re.IGNORECASE | re.DOTALL)
            for p in (step_patterns or self.DEFAULT_STEP_PATTERNS)
        ]
        self._conclusion_patterns: List[Pattern] = [
            re.compile(p, re.IGNORECASE | re.DOTALL)
            for p in (conclusion_patterns or self.DEFAULT_CONCLUSION_PATTERNS)
        ]
        self.log_every = log_every
        self._traces: List[CoTTrace] = []
        self._update_count: int = 0

    def extract(
        self,
        text: str,
        sample_id: Optional[Any] = None,
    ) -> CoTTrace:
        """
        Extract a CoT trace from a single model output string.

        Args:
            text: Raw model output text.
            sample_id: Optional identifier for this sample.

        Returns:
            A :class:`CoTTrace` containing extracted steps.
        """
        steps: List[CoTStep] = []
        step_index = 0

        for pattern in self._step_patterns:
            for match in pattern.finditer(text):
                step_text = match.group(1).strip() if match.lastindex else match.group(0).strip()
                if step_text:
                    steps.append(
                        CoTStep(index=step_index, text=step_text, is_conclusion=False)
                    )
                    step_index += 1

        # Only attach a conclusion when at least one reasoning step was found.
        # This prevents bare "The answer is X." sentences from being flagged as CoT.
        if steps:
            for pattern in self._conclusion_patterns:
                match = pattern.search(text)
                if match:
                    conclusion_text = match.group(1).strip() if match.lastindex else match.group(0).strip()
                    if conclusion_text:
                        steps.append(
                            CoTStep(index=step_index, text=conclusion_text, is_conclusion=True)
                        )
                    break

        return CoTTrace(
            raw_text=text,
            steps=steps,
            has_cot=len(steps) > 0,
            sample_id=sample_id,
        )

    def update(
        self,
        texts: List[str],
        sample_ids: Optional[List[Any]] = None,
    ) -> List[CoTTrace]:
        """
        Extract CoT traces from a batch of model outputs and accumulate state.

        Args:
            texts: List of model output strings.
            sample_ids: Optional list of sample identifiers.

        Returns:
            List of extracted :class:`CoTTrace` objects.
        """
        ids = sample_ids or [None] * len(texts)
        traces = [self.extract(t, sid) for t, sid in zip(texts, ids)]
        self._traces.extend(traces)
        self._update_count += 1

        if self.log_every > 0 and self._update_count % self.log_every == 0:
            self._log_stats()

        return traces

    def get_stats(self) -> Dict[str, Any]:
        """
        Return aggregated statistics over all accumulated traces.

        Returns:
            Dictionary with total_samples, cot_detected, cot_rate, avg_steps.
        """
        total = len(self._traces)
        if total == 0:
            return {"total_samples": 0, "cot_detected": 0, "cot_rate": 0.0, "avg_steps": 0.0}

        cot_detected = sum(1 for t in self._traces if t.has_cot)
        avg_steps = sum(t.step_count for t in self._traces) / total

        return {
            "total_samples": total,
            "cot_detected": cot_detected,
            "cot_rate": round(cot_detected / total, 4),
            "avg_steps": round(avg_steps, 2),
        }

    def reset(self) -> None:
        """Clear all accumulated traces and counters."""
        self._traces = []
        self._update_count = 0

    def _log_stats(self) -> None:
        stats = self.get_stats()
        logger.info(
            "[CoTMonitor] update=%d | cot_rate=%.3f | avg_steps=%.2f",
            self._update_count,
            stats["cot_rate"],
            stats["avg_steps"],
        )
