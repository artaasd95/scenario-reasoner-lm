"""
Dataset evaluation utilities for Scenario Reasoner LM.

Provides tools for assessing the quality, diversity, and completeness of
datasets used in scenario-based reasoning training.
"""

from __future__ import annotations

import logging
from collections import Counter
from typing import Any, Callable, Dict, List, Optional

from src.data.base_dataset import BaseScenarioDataset

logger = logging.getLogger(__name__)


class DatasetEvaluator:
    """
    Evaluates the quality and characteristics of a :class:`BaseScenarioDataset`.

    Example::

        evaluator = DatasetEvaluator(dataset)
        report    = evaluator.evaluate()
        print(report)

    Args:
        dataset: The dataset instance to evaluate.
    """

    def __init__(self, dataset: BaseScenarioDataset) -> None:
        self.dataset = dataset

    def evaluate(self) -> Dict[str, Any]:
        """
        Run all built-in evaluation checks and return a combined report.

        Returns:
            Dictionary with size, split, label distribution, input length stats,
            and reasoning trace information.
        """
        report: Dict[str, Any] = {
            "size": len(self.dataset),
            "split": self.dataset.split,
        }

        samples = [self.dataset[i] for i in range(min(len(self.dataset), 10_000))]

        report["label_distribution"] = self._label_distribution(samples)
        report["input_length_stats"] = self._length_stats(
            [str(s.get("input", "")) for s in samples]
        )
        report["has_reasoning_traces"] = any(
            "reasoning_trace" in s for s in samples
        )
        if report["has_reasoning_traces"]:
            report["trace_length_stats"] = self._length_stats(
                [str(s.get("reasoning_trace", "")) for s in samples
                 if "reasoning_trace" in s]
            )

        return report

    def label_distribution(self) -> Dict[str, int]:
        """Count the frequency of each unique label in the dataset."""
        samples = [self.dataset[i] for i in range(len(self.dataset))]
        return self._label_distribution(samples)

    def input_length_stats(self) -> Dict[str, float]:
        """Compute character-length statistics for input fields."""
        texts = [str(self.dataset[i].get("input", "")) for i in range(len(self.dataset))]
        return self._length_stats(texts)

    def coverage(
        self,
        vocab_fn: Optional[Callable[[str], List[str]]] = None,
    ) -> Dict[str, Any]:
        """
        Estimate vocabulary coverage (type-token ratio) of input texts.

        Args:
            vocab_fn: Optional tokeniser. If ``None``, whitespace splitting is used.

        Returns:
            Dictionary with unique_tokens, total_tokens, and type_token_ratio.
        """
        tokenize: Callable[[str], List[str]] = vocab_fn or str.split
        total_tokens: List[str] = []
        for i in range(len(self.dataset)):
            text = str(self.dataset[i].get("input", ""))
            total_tokens.extend(tokenize(text))

        unique = len(set(total_tokens))
        total = len(total_tokens)
        ttr = unique / total if total > 0 else 0.0
        return {
            "unique_tokens": unique,
            "total_tokens": total,
            "type_token_ratio": round(ttr, 4),
        }

    def check_missing_fields(
        self, required_fields: List[str]
    ) -> Dict[str, int]:
        """
        Count samples that are missing one or more required fields.

        Args:
            required_fields: List of field names that each sample must have.

        Returns:
            Dictionary mapping each field to the count of samples where it is absent.
        """
        missing: Dict[str, int] = {field: 0 for field in required_fields}
        for i in range(len(self.dataset)):
            sample = self.dataset[i]
            for field in required_fields:
                if field not in sample:
                    missing[field] += 1
        return missing

    @staticmethod
    def _label_distribution(samples: List[Dict[str, Any]]) -> Dict[str, int]:
        labels = [str(s["label"]) for s in samples if "label" in s]
        if not labels:
            return {}
        return dict(Counter(labels))

    @staticmethod
    def _length_stats(texts: List[str]) -> Dict[str, float]:
        if not texts:
            return {"min": 0.0, "max": 0.0, "mean": 0.0, "median": 0.0}
        lengths = sorted(len(t) for t in texts)
        n = len(lengths)
        mid = n // 2
        median = (
            lengths[mid]
            if n % 2 == 1
            else (lengths[mid - 1] + lengths[mid]) / 2
        )
        return {
            "min": float(lengths[0]),
            "max": float(lengths[-1]),
            "mean": round(sum(lengths) / n, 2),
            "median": float(median),
        }
