"""
Aha-moment catcher for Scenario Reasoner LM.

Detects sudden "aha moments" — signals of novel insight or breakthrough
understanding — in LLM reasoning traces.

Aha moments are identified through:
* Explicit insight markers (``"I see now"``, ``"Wait, that means..."``)
* Contrastive correction (``"Actually, I was wrong — ..."``)
* Confidence escalation patterns
* Domain-specific breakthrough keywords
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Pattern

logger = logging.getLogger(__name__)


@dataclass
class AhaMoment:
    """A single aha-moment event detected in a reasoning trace."""

    index: int
    text: str
    trigger_pattern: str
    position: int
    confidence: float = 1.0


@dataclass
class AhaTrace:
    """All aha moments detected in one model output."""

    raw_text: str
    moments: List[AhaMoment] = field(default_factory=list)
    has_aha: bool = False
    moment_count: int = 0
    sample_id: Optional[Any] = None

    def __post_init__(self) -> None:
        self.moment_count = len(self.moments)
        self.has_aha = self.moment_count > 0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "has_aha": self.has_aha,
            "moment_count": self.moment_count,
            "sample_id": self.sample_id,
            "moments": [
                {
                    "index": m.index,
                    "text": m.text,
                    "trigger_pattern": m.trigger_pattern,
                    "position": m.position,
                    "confidence": m.confidence,
                }
                for m in self.moments
            ],
        }


class AhaMonitor:
    """
    Catches "aha moments" — signals of sudden insight — in LLM outputs.

    Detected signals include:
    * Explicit insight declarations: *"I see now"*, *"Aha!"*, *"Of course!"*
    * Self-correction: *"Wait, actually ..."*, *"I was wrong — ..."*
    * Realisation transitions: *"This means that ..."*, *"That's why ..."*
    * Confidence escalation: *"This must be ..."*, *"It's clear that ..."*

    Example::

        monitor = AhaMonitor()
        text = "Wait, actually that changes everything. I see now — the answer is X."
        trace = monitor.extract(text)
        print(trace.moment_count)

    Args:
        patterns: List of ``(pattern_string, label)`` tuples.
            If ``None``, the default library is used.
        context_window: Number of characters to capture around each match.
        log_every: Log stats every *n* calls to :meth:`update`.
    """

    DEFAULT_PATTERNS: List[tuple] = [
        (r"\b(?:Aha|Eureka|Oh|I see)[!,]?\b", "explicit_insight"),
        (r"\bOf course[!,]?\b", "explicit_insight"),
        (r"\b(?:I(?:'ve)? (?:got it|understand now|see now|realise now|realize now))[!,]?\b", "explicit_insight"),
        (r"\bWait[,!]?\s+(?:actually|no|but|that means|this means)\b", "self_correction"),
        (r"\b(?:Actually|But wait)[,!]?\s+(?:I was wrong|that's not right|this changes)\b", "self_correction"),
        (r"\b(?:This means|That means|So it follows|Therefore|This implies)\b.{0,40}!", "realisation"),
        (r"\b(?:That's why|That explains|Now I understand)\b", "realisation"),
        (r"\b(?:It must be|This must be|It's clear that|Clearly|Obviously)\b", "confidence_escalation"),
        (r"\b(?:The key insight|The crucial point|The real issue)[,:]?\s+(?:is|was)\b", "key_insight"),
    ]

    def __init__(
        self,
        patterns: Optional[List[tuple]] = None,
        context_window: int = 120,
        log_every: int = 100,
    ) -> None:
        raw_patterns = patterns or self.DEFAULT_PATTERNS
        self._compiled: List[tuple] = [
            (re.compile(p, re.IGNORECASE), label)
            for p, label in raw_patterns
        ]
        self.context_window = context_window
        self.log_every = log_every
        self._traces: List[AhaTrace] = []
        self._update_count: int = 0

    def extract(
        self,
        text: str,
        sample_id: Optional[Any] = None,
    ) -> AhaTrace:
        """
        Detect aha moments in a single model output string.

        Args:
            text: Raw model output text.
            sample_id: Optional identifier for this sample.

        Returns:
            An :class:`AhaTrace` with all detected moments.
        """
        moments: List[AhaMoment] = []
        moment_index = 0
        seen_positions: set = set()

        for pattern, label in self._compiled:
            for match in pattern.finditer(text):
                pos = match.start()
                if any(abs(pos - p) < 30 for p in seen_positions):
                    continue
                seen_positions.add(pos)

                start = max(0, pos - self.context_window // 2)
                end = min(len(text), pos + self.context_window // 2)
                context = text[start:end].strip()

                moments.append(
                    AhaMoment(
                        index=moment_index,
                        text=context,
                        trigger_pattern=label,
                        position=pos,
                    )
                )
                moment_index += 1

        moments.sort(key=lambda m: m.position)
        for i, m in enumerate(moments):
            m.index = i

        return AhaTrace(
            raw_text=text,
            moments=moments,
            sample_id=sample_id,
        )

    def update(
        self,
        texts: List[str],
        sample_ids: Optional[List[Any]] = None,
    ) -> List[AhaTrace]:
        """
        Extract aha moments from a batch and accumulate state.

        Args:
            texts: List of model output strings.
            sample_ids: Optional list of sample identifiers.

        Returns:
            List of :class:`AhaTrace` objects.
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
            Dictionary with total_samples, aha_detected, aha_rate,
            avg_moments, pattern_breakdown.
        """
        total = len(self._traces)
        if total == 0:
            return {
                "total_samples": 0,
                "aha_detected": 0,
                "aha_rate": 0.0,
                "avg_moments": 0.0,
                "pattern_breakdown": {},
            }

        aha_detected = sum(1 for t in self._traces if t.has_aha)
        avg_moments = sum(t.moment_count for t in self._traces) / total

        breakdown: Dict[str, int] = {}
        for trace in self._traces:
            for m in trace.moments:
                breakdown[m.trigger_pattern] = breakdown.get(m.trigger_pattern, 0) + 1

        return {
            "total_samples": total,
            "aha_detected": aha_detected,
            "aha_rate": round(aha_detected / total, 4),
            "avg_moments": round(avg_moments, 2),
            "pattern_breakdown": breakdown,
        }

    def reset(self) -> None:
        """Clear all accumulated traces and counters."""
        self._traces = []
        self._update_count = 0

    def _log_stats(self) -> None:
        stats = self.get_stats()
        logger.info(
            "[AhaMonitor] update=%d | aha_rate=%.3f | avg_moments=%.2f",
            self._update_count,
            stats["aha_rate"],
            stats["avg_moments"],
        )
