"""
Tree-of-Thought (ToT) monitor for Scenario Reasoner LM.

Detects, extracts, and logs tree-of-thought reasoning patterns from
LLM output text.  ToT reasoning is characterised by the model exploring
multiple branches or hypotheses and then evaluating / pruning them.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Pattern

logger = logging.getLogger(__name__)


@dataclass
class ThoughtNode:
    """A single node in a Tree-of-Thought exploration."""

    node_id: int
    text: str
    parent_id: Optional[int] = None
    score: Optional[float] = None
    is_selected: bool = False
    is_pruned: bool = False


@dataclass
class ToTTrace:
    """A complete Tree-of-Thought trace extracted from one model output."""

    raw_text: str
    nodes: List[ThoughtNode] = field(default_factory=list)
    has_tot: bool = False
    branch_count: int = 0
    selected_path: List[int] = field(default_factory=list)
    sample_id: Optional[Any] = None

    def __post_init__(self) -> None:
        self.branch_count = len(self.nodes)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "has_tot": self.has_tot,
            "branch_count": self.branch_count,
            "selected_path": self.selected_path,
            "sample_id": self.sample_id,
            "nodes": [
                {
                    "node_id": n.node_id,
                    "text": n.text,
                    "parent_id": n.parent_id,
                    "score": n.score,
                    "is_selected": n.is_selected,
                    "is_pruned": n.is_pruned,
                }
                for n in self.nodes
            ],
        }


class ToTMonitor:
    """
    Monitors and extracts Tree-of-Thought (ToT) patterns from LLM outputs.

    ToT reasoning is typically marked by:
    * Explicit branch / option enumeration (``"Option A: ..."`` / ``"Hypothesis 1: ..."``)
    * Evaluation or scoring of candidates
    * Pruning (``"Discarding option ..."``).
    * Selection (``"Best option: ..."``)

    Example::

        monitor = ToTMonitor()
        trace   = monitor.extract("Option 1: ... Option 2: ... Best option: Option 2.")
        print(trace.branch_count)

    Args:
        branch_patterns: Regex patterns identifying new branches/options.
        evaluation_patterns: Patterns signalling evaluation/scoring.
        pruning_patterns: Patterns signalling pruning.
        selection_patterns: Patterns signalling the selection of a branch.
        log_every: Log stats every *n* calls to :meth:`update`.
    """

    DEFAULT_BRANCH_PATTERNS: List[str] = [
        r"(?:Option|Hypothesis|Path|Branch|Approach|Candidate)\s*(?:\d+|[A-Za-z])[:\.\)]\s*(.+?)(?=(?:Option|Hypothesis|Path|Branch|Approach|Candidate)\s*(?:\d+|[A-Za-z])[:\.\)]|$)",
        r"(?:Let me try|What if|Another approach|Alternatively)[:\,]?\s*(.+?)(?=(?:Let me try|What if|Another approach|Alternatively|Therefore|In conclusion)|$)",
    ]

    DEFAULT_EVALUATION_PATTERNS: List[str] = [
        r"(?:Score|Rating|Evaluation)[:\s]+([0-9.]+)",
        r"(?:This is|This seems|This appears)\s+(?:good|better|best|promising|correct|valid)",
    ]

    DEFAULT_PRUNING_PATTERNS: List[str] = [
        r"(?:Discarding|Pruning|Eliminating|Ruling out|This doesn't work)[:\,]?\s*(.+?)(?=\.|$)",
        r"(?:This is|This seems|This appears)\s+(?:wrong|incorrect|invalid|unlikely|not valid)",
    ]

    DEFAULT_SELECTION_PATTERNS: List[str] = [
        r"(?:Best option|Choosing|I choose|Selected path|Final choice)[:\,]?\s*(.+?)(?=\.|$)",
        r"(?:Therefore|Thus|The best answer)[,:]?\s*(.+)",
    ]

    def __init__(
        self,
        branch_patterns: Optional[List[str]] = None,
        evaluation_patterns: Optional[List[str]] = None,
        pruning_patterns: Optional[List[str]] = None,
        selection_patterns: Optional[List[str]] = None,
        log_every: int = 100,
    ) -> None:
        def _compile(patterns: List[str]) -> List[Pattern]:
            return [re.compile(p, re.IGNORECASE | re.DOTALL) for p in patterns]

        self._branch_patterns = _compile(branch_patterns or self.DEFAULT_BRANCH_PATTERNS)
        self._evaluation_patterns = _compile(evaluation_patterns or self.DEFAULT_EVALUATION_PATTERNS)
        self._pruning_patterns = _compile(pruning_patterns or self.DEFAULT_PRUNING_PATTERNS)
        self._selection_patterns = _compile(selection_patterns or self.DEFAULT_SELECTION_PATTERNS)

        self.log_every = log_every
        self._traces: List[ToTTrace] = []
        self._update_count: int = 0

    def extract(
        self,
        text: str,
        sample_id: Optional[Any] = None,
    ) -> ToTTrace:
        """
        Extract a ToT trace from a single model output string.

        Args:
            text: Raw model output text.
            sample_id: Optional identifier for this sample.

        Returns:
            A :class:`ToTTrace` with detected thought nodes.
        """
        nodes: List[ThoughtNode] = []
        node_id = 0
        selected_path: List[int] = []

        for pattern in self._branch_patterns:
            for match in pattern.finditer(text):
                branch_text = match.group(1).strip() if match.lastindex else match.group(0).strip()
                if branch_text:
                    node = ThoughtNode(node_id=node_id, text=branch_text)
                    for prune_pat in self._pruning_patterns:
                        if prune_pat.search(branch_text):
                            node.is_pruned = True
                            break
                    nodes.append(node)
                    node_id += 1

        for pattern in self._selection_patterns:
            match = pattern.search(text)
            if match:
                for n in nodes:
                    if not n.is_pruned:
                        n.is_selected = True
                        selected_path.append(n.node_id)
                break

        return ToTTrace(
            raw_text=text,
            nodes=nodes,
            has_tot=len(nodes) > 1,
            selected_path=selected_path,
            sample_id=sample_id,
        )

    def update(
        self,
        texts: List[str],
        sample_ids: Optional[List[Any]] = None,
    ) -> List[ToTTrace]:
        """
        Extract ToT traces from a batch and accumulate state.

        Args:
            texts: List of model output strings.
            sample_ids: Optional list of sample identifiers.

        Returns:
            List of :class:`ToTTrace` objects.
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
            Dictionary with total_samples, tot_detected, tot_rate,
            avg_branches, avg_pruned.
        """
        total = len(self._traces)
        if total == 0:
            return {
                "total_samples": 0,
                "tot_detected": 0,
                "tot_rate": 0.0,
                "avg_branches": 0.0,
                "avg_pruned": 0.0,
            }

        tot_detected = sum(1 for t in self._traces if t.has_tot)
        avg_branches = sum(t.branch_count for t in self._traces) / total
        avg_pruned = (
            sum(sum(1 for n in t.nodes if n.is_pruned) for t in self._traces) / total
        )

        return {
            "total_samples": total,
            "tot_detected": tot_detected,
            "tot_rate": round(tot_detected / total, 4),
            "avg_branches": round(avg_branches, 2),
            "avg_pruned": round(avg_pruned, 2),
        }

    def reset(self) -> None:
        """Clear all accumulated traces and counters."""
        self._traces = []
        self._update_count = 0

    def _log_stats(self) -> None:
        stats = self.get_stats()
        logger.info(
            "[ToTMonitor] update=%d | tot_rate=%.3f | avg_branches=%.2f | avg_pruned=%.2f",
            self._update_count,
            stats["tot_rate"],
            stats["avg_branches"],
            stats["avg_pruned"],
        )
