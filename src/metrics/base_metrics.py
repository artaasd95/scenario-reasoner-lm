"""
Base metric class hierarchy for Scenario Reasoner LM.

All evaluation metrics should inherit from :class:`BaseMetric`.
A lightweight :class:`MetricRegistry` is provided so trainers can manage
multiple metrics by name.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional


class BaseMetric(ABC):
    """
    Abstract base class for scenario reasoning evaluation metrics.

    Metrics follow an *accumulate-then-compute* pattern:

    1. Call :meth:`update` for each batch of predictions/references.
    2. Call :meth:`compute` to obtain the aggregated metric value.
    3. Call :meth:`reset` to clear accumulated state before the next epoch.

    Example::

        class ExactMatch(BaseMetric):
            name = "exact_match"

            def __init__(self):
                super().__init__()
                self._correct = 0
                self._total   = 0

            def update(self, predictions, references):
                self._correct += sum(p == r for p, r in zip(predictions, references))
                self._total   += len(predictions)

            def compute(self):
                return self._correct / self._total if self._total else 0.0

    Attributes:
        name: Human-readable metric identifier.
    """

    name: str = "base_metric"

    def __init__(self) -> None:
        self._state: Dict[str, Any] = {}
        self.reset()

    @abstractmethod
    def update(self, predictions: Any, references: Any, **kwargs: Any) -> None:
        """
        Accumulate metric state from a new batch.

        Args:
            predictions: Model outputs for the current batch.
            references: Ground-truth targets for the current batch.
        """

    @abstractmethod
    def compute(self) -> Any:
        """
        Compute and return the aggregated metric value.

        Returns:
            Metric value (typically a float or dict).
        """

    def reset(self) -> None:
        """Reset accumulated state."""
        self._state = {}

    def __call__(self, predictions: Any, references: Any, **kwargs: Any) -> Any:
        """
        Convenience: reset, update with one batch, and compute.

        Args:
            predictions: Model outputs.
            references: Ground-truth targets.

        Returns:
            Metric value for this batch.
        """
        self.reset()
        self.update(predictions, references, **kwargs)
        return self.compute()

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(name={self.name!r})"


class MetricRegistry:
    """
    A simple registry for managing multiple :class:`BaseMetric` instances.

    Example::

        registry = MetricRegistry()
        registry.register(ExactMatch())
        registry.register(RougeMetric())

        for batch in dataloader:
            preds, refs = model(batch), batch["labels"]
            registry.update_all(preds, refs)

        results = registry.compute_all()
        registry.reset_all()

    Args:
        metrics: Optional list of :class:`BaseMetric` instances to pre-register.
    """

    def __init__(self, metrics: Optional[List[BaseMetric]] = None) -> None:
        self._metrics: Dict[str, BaseMetric] = {}
        for metric in metrics or []:
            self.register(metric)

    def register(self, metric: BaseMetric) -> None:
        """
        Add a metric to the registry.

        Args:
            metric: A :class:`BaseMetric` instance.

        Raises:
            ValueError: If a metric with the same name is already registered.
        """
        if metric.name in self._metrics:
            raise ValueError(
                f"A metric named '{metric.name}' is already registered."
            )
        self._metrics[metric.name] = metric

    def unregister(self, name: str) -> None:
        """
        Remove a metric from the registry.

        Args:
            name: Name of the metric to remove.

        Raises:
            KeyError: If no metric with that name exists.
        """
        if name not in self._metrics:
            raise KeyError(f"No metric named '{name}' in registry.")
        del self._metrics[name]

    def get(self, name: str) -> BaseMetric:
        """
        Retrieve a registered metric by name.

        Raises:
            KeyError: If no metric with that name exists.
        """
        if name not in self._metrics:
            raise KeyError(f"No metric named '{name}' in registry.")
        return self._metrics[name]

    def update_all(self, predictions: Any, references: Any, **kwargs: Any) -> None:
        """Call :meth:`~BaseMetric.update` on every registered metric."""
        for metric in self._metrics.values():
            metric.update(predictions, references, **kwargs)

    def compute_all(self) -> Dict[str, Any]:
        """Compute and return results from all registered metrics."""
        return {name: metric.compute() for name, metric in self._metrics.items()}

    def reset_all(self) -> None:
        """Reset all registered metrics."""
        for metric in self._metrics.values():
            metric.reset()

    def __contains__(self, name: str) -> bool:
        return name in self._metrics

    def __len__(self) -> int:
        return len(self._metrics)

    def __repr__(self) -> str:
        names = list(self._metrics.keys())
        return f"MetricRegistry(metrics={names})"
