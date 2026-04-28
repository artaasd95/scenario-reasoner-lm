"""
Custom dataset implementations for Scenario Reasoner LM.

Provides concrete PyTorch Dataset classes for scenario-based reasoning tasks,
backed by in-memory tensors or raw Python lists.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple, Union

import torch

from src.data.base_dataset import BaseScenarioDataset


class ScenarioDataset(BaseScenarioDataset):
    """
    A general-purpose scenario dataset backed by parallel lists or tensors.

    Each sample represents one *scenario* — a structured context that a model
    is expected to reason about.

    Example::

        scenarios = ["Scenario A: ...", "Scenario B: ..."]
        labels    = ["correct", "incorrect"]
        dataset   = ScenarioDataset(scenarios, labels=labels, split="train")

    Args:
        scenarios: List of scenario strings or feature tensors.
        labels: Optional list of target labels.
        reasoning_traces: Optional list of ground-truth reasoning trace strings.
        split: Dataset split identifier.
        metadata: Optional dataset-level metadata dictionary.
    """

    def __init__(
        self,
        scenarios: List[Union[str, torch.Tensor]],
        labels: Optional[List[Any]] = None,
        reasoning_traces: Optional[List[str]] = None,
        split: str = "train",
        metadata: Optional[Dict[str, Any]] = None,
    ) -> None:
        super().__init__(split=split, metadata=metadata)

        if labels is not None and len(labels) != len(scenarios):
            raise ValueError(
                f"Length mismatch: {len(scenarios)} scenarios but {len(labels)} labels."
            )
        if reasoning_traces is not None and len(reasoning_traces) != len(scenarios):
            raise ValueError(
                f"Length mismatch: {len(scenarios)} scenarios but "
                f"{len(reasoning_traces)} reasoning_traces."
            )

        self._scenarios = scenarios
        self._labels = labels
        self._reasoning_traces = reasoning_traces

    def __len__(self) -> int:
        return len(self._scenarios)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        sample: Dict[str, Any] = {"input": self._scenarios[idx], "idx": idx}

        if self._labels is not None:
            sample["label"] = self._labels[idx]

        if self._reasoning_traces is not None:
            sample["reasoning_trace"] = self._reasoning_traces[idx]

        return self.preprocess(sample)


class ReasoningTraceDataset(BaseScenarioDataset):
    """
    Dataset built from (input, reasoning_trace, output) triplets.

    Designed for fine-tuning or evaluation workflows where the full
    chain-of-thought is available as a training signal.

    Example::

        inputs  = ["What is 2 + 2?", "Is the sky blue?"]
        traces  = ["Let me think step by step. 2 + 2 = 4.", "Yes, by Rayleigh scattering."]
        outputs = ["4", "Yes"]
        dataset = ReasoningTraceDataset(inputs, traces, outputs)

    Args:
        inputs: List of input strings (questions, prompts, scenarios).
        reasoning_traces: List of step-by-step reasoning trace strings.
        outputs: List of final answer / output strings.
        split: Dataset split identifier.
        metadata: Optional dataset-level metadata dictionary.
    """

    def __init__(
        self,
        inputs: List[str],
        reasoning_traces: List[str],
        outputs: List[str],
        split: str = "train",
        metadata: Optional[Dict[str, Any]] = None,
    ) -> None:
        super().__init__(split=split, metadata=metadata)

        if not (len(inputs) == len(reasoning_traces) == len(outputs)):
            raise ValueError(
                "inputs, reasoning_traces, and outputs must all have the same length."
            )

        self._inputs = inputs
        self._reasoning_traces = reasoning_traces
        self._outputs = outputs

    def __len__(self) -> int:
        return len(self._inputs)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        sample = {
            "input": self._inputs[idx],
            "reasoning_trace": self._reasoning_traces[idx],
            "output": self._outputs[idx],
            "idx": idx,
        }
        return self.preprocess(sample)

    def as_pairs(self) -> List[Tuple[str, str]]:
        """Return a list of ``(input, output)`` tuples (trace excluded)."""
        return list(zip(self._inputs, self._outputs))

    def as_triplets(self) -> List[Tuple[str, str, str]]:
        """Return a list of ``(input, trace, output)`` triplets."""
        return list(zip(self._inputs, self._reasoning_traces, self._outputs))
