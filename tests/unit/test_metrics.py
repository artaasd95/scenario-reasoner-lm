"""
Unit tests for base metric classes.
"""

import pytest
from typing import Any, List

from src.metrics.base_metrics import BaseMetric, MetricRegistry


# ---------------------------------------------------------------------------
# Concrete metric helpers
# ---------------------------------------------------------------------------

class _ExactMatch(BaseMetric):
    name = "exact_match"

    def __init__(self):
        super().__init__()
        self._correct = 0
        self._total = 0

    def update(self, predictions: List[str], references: List[str], **kwargs: Any) -> None:
        self._correct += sum(p == r for p, r in zip(predictions, references))
        self._total += len(predictions)

    def compute(self) -> float:
        return self._correct / self._total if self._total else 0.0

    def reset(self) -> None:
        super().reset()
        self._correct = 0
        self._total = 0


class _AlwaysOne(BaseMetric):
    name = "always_one"

    def update(self, predictions: Any, references: Any, **kwargs: Any) -> None:
        pass

    def compute(self) -> float:
        return 1.0


# ---------------------------------------------------------------------------
# BaseMetric tests
# ---------------------------------------------------------------------------

class TestBaseMetric:
    def test_abstract_cannot_instantiate(self):
        with pytest.raises(TypeError):
            BaseMetric()  # type: ignore[abstract]

    def test_update_compute_reset(self):
        m = _ExactMatch()
        m.update(["a", "b"], ["a", "c"])
        assert m.compute() == pytest.approx(0.5)
        m.reset()
        assert m.compute() == 0.0

    def test_callable_interface(self):
        m = _ExactMatch()
        score = m(["x", "x"], ["x", "y"])
        assert score == pytest.approx(0.5)

    def test_callable_resets_state(self):
        m = _ExactMatch()
        m.update(["a"], ["a"])  # accumulate something first
        score = m(["x"], ["z"])  # __call__ should reset then compute
        assert score == pytest.approx(0.0)

    def test_repr(self):
        m = _ExactMatch()
        assert "exact_match" in repr(m)


# ---------------------------------------------------------------------------
# MetricRegistry tests
# ---------------------------------------------------------------------------

class TestMetricRegistry:
    def test_len(self):
        reg = MetricRegistry()
        reg.register(_ExactMatch())
        assert len(reg) == 1

    def test_contains(self):
        reg = MetricRegistry()
        reg.register(_ExactMatch())
        assert "exact_match" in reg
        assert "other" not in reg

    def test_get(self):
        reg = MetricRegistry()
        m = _ExactMatch()
        reg.register(m)
        assert reg.get("exact_match") is m

    def test_get_missing_raises(self):
        reg = MetricRegistry()
        with pytest.raises(KeyError):
            reg.get("nonexistent")

    def test_register_duplicate_raises(self):
        reg = MetricRegistry()
        reg.register(_ExactMatch())
        with pytest.raises(ValueError):
            reg.register(_ExactMatch())

    def test_unregister(self):
        reg = MetricRegistry()
        reg.register(_ExactMatch())
        reg.unregister("exact_match")
        assert "exact_match" not in reg

    def test_unregister_missing_raises(self):
        reg = MetricRegistry()
        with pytest.raises(KeyError):
            reg.unregister("nope")

    def test_update_all_compute_all(self):
        reg = MetricRegistry(metrics=[_ExactMatch(), _AlwaysOne()])
        reg.update_all(["a"], ["a"])
        results = reg.compute_all()
        assert results["exact_match"] == pytest.approx(1.0)
        assert results["always_one"] == pytest.approx(1.0)

    def test_reset_all(self):
        reg = MetricRegistry(metrics=[_ExactMatch()])
        reg.update_all(["a"], ["a"])
        reg.reset_all()
        assert reg.get("exact_match").compute() == 0.0

    def test_init_with_list(self):
        reg = MetricRegistry(metrics=[_ExactMatch(), _AlwaysOne()])
        assert len(reg) == 2

    def test_repr(self):
        reg = MetricRegistry(metrics=[_ExactMatch()])
        r = repr(reg)
        assert "exact_match" in r
