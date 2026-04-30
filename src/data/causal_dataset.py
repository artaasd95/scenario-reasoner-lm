"""
Causal reasoning dataset — extends ReasoningTraceDataset with θ metadata.

Each sample carries its :class:`~src.scenarios.causal.taxonomy.CausalTheta`
parameter vector so downstream code (metrics, robustness evaluation) can
perform per-θ breakdowns without reprocessing the raw scenario data.
"""

from __future__ import annotations

import random
from typing import Any, Dict, List, Optional, Tuple

from src.data.custom_datasets import ReasoningTraceDataset


class CausalReasoningDataset(ReasoningTraceDataset):
    """
    Dataset of causal/counterfactual reasoning triplets with θ metadata.

    Each sample is a dict with keys:
        ``input``           — scenario prompt
        ``reasoning_trace`` — CoT ground-truth trace
        ``output``          — expected answer
        ``theta``           — :class:`CausalTheta` as a plain dict

    Args:
        inputs: List of scenario prompt strings.
        reasoning_traces: List of CoT reasoning trace strings.
        outputs: List of expected answer strings.
        thetas: Optional list of :class:`CausalTheta` objects or dicts.
            Defaults to empty dicts when omitted.
        split: Dataset split identifier (``"train"``, ``"val"``, ``"test"``).
        metadata: Optional dataset-level metadata dict.

    Example::

        dataset = CausalReasoningDataset(inputs, traces, outputs, thetas)
        sample  = dataset[0]
        print(sample["theta"]["domain"])

        train_ds, val_ds = dataset.stratified_split(train_frac=0.9)

        physical_ds = dataset.filter_by_theta("domain", "physical")
    """

    def __init__(
        self,
        inputs: List[str],
        reasoning_traces: List[str],
        outputs: List[str],
        thetas: Optional[List[Any]] = None,
        split: str = "train",
        metadata: Optional[Dict[str, Any]] = None,
    ) -> None:
        super().__init__(
            inputs=inputs,
            reasoning_traces=reasoning_traces,
            outputs=outputs,
            split=split,
            metadata=metadata,
        )
        self._thetas: List[Any] = thetas if thetas is not None else [{}] * len(inputs)
        if len(self._thetas) != len(inputs):
            raise ValueError(
                f"Length mismatch: {len(inputs)} inputs but {len(self._thetas)} thetas."
            )

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        sample = super().__getitem__(idx)
        theta = self._thetas[idx]
        sample["theta"] = theta.to_dict() if hasattr(theta, "to_dict") else theta
        return sample

    # ── Filtering helpers ────────────────────────────────────────────────────

    def filter_by_theta(self, field: str, value: Any) -> "CausalReasoningDataset":
        """
        Return a new dataset containing only samples whose θ ``field == value``.

        Args:
            field: Name of the :class:`CausalTheta` attribute to filter on
                   (e.g. ``"domain"``, ``"difficulty"``, ``"intervention_type"``).
            value: Value to match.

        Returns:
            A new :class:`CausalReasoningDataset` with matching samples only.
        """
        indices = [
            i for i, theta in enumerate(self._thetas)
            if (
                getattr(theta, field, None) == value
                if hasattr(theta, field)
                else (theta.get(field) == value if isinstance(theta, dict) else False)
            )
        ]
        return self._subset(indices, split=self.split)

    def stratified_split(
        self,
        train_frac: float = 0.9,
        seed: int = 42,
    ) -> Tuple["CausalReasoningDataset", "CausalReasoningDataset"]:
        """
        Split the dataset into train and validation subsets.

        Stratification is performed over ``(intervention_type, domain, difficulty)``
        to ensure each θ combination is represented in both splits.

        Args:
            train_frac: Fraction of samples assigned to the training split.
            seed: Random seed for reproducibility.

        Returns:
            Tuple ``(train_dataset, val_dataset)``.
        """
        rng = random.Random(seed)

        # Group indices by stratum key
        strata: Dict[Any, List[int]] = {}
        for i, theta in enumerate(self._thetas):
            key = self._theta_stratum_key(theta)
            strata.setdefault(key, []).append(i)

        train_idx: List[int] = []
        val_idx: List[int] = []

        for indices in strata.values():
            shuffled = list(indices)
            rng.shuffle(shuffled)
            n_train = max(1, round(len(shuffled) * train_frac))
            train_idx.extend(shuffled[:n_train])
            val_idx.extend(shuffled[n_train:])

        return self._subset(train_idx, "train"), self._subset(val_idx, "val")

    # ── Factory method ───────────────────────────────────────────────────────

    @classmethod
    def from_scenario_instances(
        cls,
        instances: List[Any],
        split: str = "train",
    ) -> "CausalReasoningDataset":
        """
        Build a :class:`CausalReasoningDataset` from a list of
        :class:`~src.scenarios.base_scenario.ScenarioInstance` objects.

        Args:
            instances: List of :class:`ScenarioInstance` produced by
                       :class:`~src.scenarios.causal.generator.CausalScenarioGenerator`.
            split: Dataset split identifier.

        Returns:
            A fully populated :class:`CausalReasoningDataset`.
        """
        inputs = [inst.prompt for inst in instances]
        traces = [inst.reasoning_trace or "" for inst in instances]
        outputs = [inst.answer or "" for inst in instances]
        thetas = [inst.theta for inst in instances]
        return cls(inputs, traces, outputs, thetas=thetas, split=split)

    # ── Internals ─────────────────────────────────────────────────────────────

    def _subset(self, indices: List[int], split: str) -> "CausalReasoningDataset":
        """Create a new CausalReasoningDataset from a list of indices."""
        return CausalReasoningDataset(
            inputs=[self._inputs[i] for i in indices],
            reasoning_traces=[self._reasoning_traces[i] for i in indices],
            outputs=[self._outputs[i] for i in indices],
            thetas=[self._thetas[i] for i in indices],
            split=split,
            metadata=self.metadata,
        )

    @staticmethod
    def _theta_stratum_key(theta: Any) -> tuple:
        """Extract a hashable stratum key from a theta (dataclass or dict)."""
        if hasattr(theta, "intervention_type"):
            return (theta.intervention_type, theta.domain, theta.difficulty)
        if isinstance(theta, dict):
            return (
                theta.get("intervention_type", ""),
                theta.get("domain", ""),
                theta.get("difficulty", ""),
            )
        return (str(theta),)
