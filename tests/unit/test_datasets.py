"""
Unit tests for dataset classes.
"""

import pytest
import torch

from src.data.base_dataset import BaseScenarioDataset
from src.data.custom_datasets import ScenarioDataset, ReasoningTraceDataset
from src.data.evaluators import DatasetEvaluator


# ---------------------------------------------------------------------------
# Helpers / fixtures
# ---------------------------------------------------------------------------

class _MinimalDataset(BaseScenarioDataset):
    """Minimal concrete subclass for testing the abstract base."""

    def __init__(self, size: int = 5, **kwargs):
        super().__init__(**kwargs)
        self._size = size

    def __len__(self):
        return self._size

    def __getitem__(self, idx):
        return {"input": f"item_{idx}", "label": idx % 2, "idx": idx}


@pytest.fixture()
def minimal_ds():
    return _MinimalDataset(size=5, split="train")


@pytest.fixture()
def scenario_ds():
    return ScenarioDataset(
        scenarios=["A", "B", "C"],
        labels=["yes", "no", "yes"],
        split="train",
    )


@pytest.fixture()
def trace_ds():
    return ReasoningTraceDataset(
        inputs=["Q1", "Q2"],
        reasoning_traces=["T1", "T2"],
        outputs=["A1", "A2"],
        split="train",
    )


# ---------------------------------------------------------------------------
# BaseScenarioDataset tests
# ---------------------------------------------------------------------------

class TestBaseScenarioDataset:
    def test_len(self, minimal_ds):
        assert len(minimal_ds) == 5

    def test_getitem(self, minimal_ds):
        sample = minimal_ds[0]
        assert "input" in sample
        assert "label" in sample

    def test_iteration(self, minimal_ds):
        items = list(minimal_ds)
        assert len(items) == 5

    def test_split_stored(self, minimal_ds):
        assert minimal_ds.split == "train"

    def test_repr(self, minimal_ds):
        r = repr(minimal_ds)
        assert "train" in r
        assert "5" in r

    def test_dataloader_created(self, minimal_ds):
        loader = minimal_ds.get_dataloader(batch_size=2)
        batch = next(iter(loader))
        assert "input" in batch

    def test_collate_fn_tensors(self, minimal_ds):
        batch = [{"x": torch.tensor([1.0]), "y": "label"}
                 for _ in range(3)]
        collated = minimal_ds.collate_fn(batch)
        assert isinstance(collated["x"], torch.Tensor)
        assert collated["x"].shape[0] == 3
        assert isinstance(collated["y"], list)


# ---------------------------------------------------------------------------
# ScenarioDataset tests
# ---------------------------------------------------------------------------

class TestScenarioDataset:
    def test_basic(self, scenario_ds):
        assert len(scenario_ds) == 3
        s = scenario_ds[0]
        assert s["input"] == "A"
        assert s["label"] == "yes"

    def test_no_labels(self):
        ds = ScenarioDataset(["X", "Y", "Z"])
        s = ds[0]
        assert "label" not in s

    def test_reasoning_traces(self):
        ds = ScenarioDataset(
            scenarios=["Q"],
            labels=["A"],
            reasoning_traces=["Step 1: ..."],
        )
        assert ds[0]["reasoning_trace"] == "Step 1: ..."

    def test_length_mismatch_raises(self):
        with pytest.raises(ValueError):
            ScenarioDataset(["A", "B"], labels=["y"])

    def test_trace_mismatch_raises(self):
        with pytest.raises(ValueError):
            ScenarioDataset(["A", "B"], reasoning_traces=["T1", "T2", "T3"])

    def test_tensor_input(self):
        t = torch.tensor([1.0, 2.0])
        ds = ScenarioDataset([t, t])
        assert isinstance(ds[0]["input"], torch.Tensor)


# ---------------------------------------------------------------------------
# ReasoningTraceDataset tests
# ---------------------------------------------------------------------------

class TestReasoningTraceDataset:
    def test_len(self, trace_ds):
        assert len(trace_ds) == 2

    def test_getitem(self, trace_ds):
        s = trace_ds[0]
        assert s["input"] == "Q1"
        assert s["reasoning_trace"] == "T1"
        assert s["output"] == "A1"

    def test_mismatch_raises(self):
        with pytest.raises(ValueError):
            ReasoningTraceDataset(["Q1"], ["T1", "T2"], ["A1"])

    def test_as_pairs(self, trace_ds):
        pairs = trace_ds.as_pairs()
        assert pairs == [("Q1", "A1"), ("Q2", "A2")]

    def test_as_triplets(self, trace_ds):
        triplets = trace_ds.as_triplets()
        assert triplets[0] == ("Q1", "T1", "A1")


# ---------------------------------------------------------------------------
# DatasetEvaluator tests
# ---------------------------------------------------------------------------

class TestDatasetEvaluator:
    def test_evaluate_returns_dict(self, scenario_ds):
        ev = DatasetEvaluator(scenario_ds)
        report = ev.evaluate()
        assert isinstance(report, dict)
        assert report["size"] == 3

    def test_label_distribution(self, scenario_ds):
        ev = DatasetEvaluator(scenario_ds)
        dist = ev.label_distribution()
        assert dist["yes"] == 2
        assert dist["no"] == 1

    def test_input_length_stats(self, scenario_ds):
        ev = DatasetEvaluator(scenario_ds)
        stats = ev.input_length_stats()
        assert "min" in stats and "max" in stats and "mean" in stats

    def test_coverage(self, scenario_ds):
        ev = DatasetEvaluator(scenario_ds)
        cov = ev.coverage()
        assert "unique_tokens" in cov
        assert "type_token_ratio" in cov

    def test_check_missing_fields(self, scenario_ds):
        ev = DatasetEvaluator(scenario_ds)
        missing = ev.check_missing_fields(["input", "label", "nonexistent"])
        assert missing["input"] == 0
        assert missing["label"] == 0
        assert missing["nonexistent"] == 3

    def test_has_reasoning_traces_false(self, scenario_ds):
        ev = DatasetEvaluator(scenario_ds)
        report = ev.evaluate()
        assert report["has_reasoning_traces"] is False

    def test_has_reasoning_traces_true(self):
        ds = ScenarioDataset(
            ["Q"], labels=["A"], reasoning_traces=["Step 1: done."]
        )
        ev = DatasetEvaluator(ds)
        report = ev.evaluate()
        assert report["has_reasoning_traces"] is True
