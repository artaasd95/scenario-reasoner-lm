"""
Integration tests — end-to-end pipeline checks.
"""

import os
import tempfile

import pytest
import torch

from src.data.custom_datasets import ScenarioDataset, ReasoningTraceDataset
from src.data.evaluators import DatasetEvaluator
from src.losses.base_losses import TokenLevelLoss, SequenceLevelLoss
from src.metrics.base_metrics import BaseMetric, MetricRegistry
from src.monitoring.cot_monitor import CoTMonitor
from src.monitoring.tot_monitor import ToTMonitor
from src.monitoring.aha_monitor import AhaMonitor
from src.logging.local_logger import LocalLogger
from typing import Any


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

class _SimpleCELoss(TokenLevelLoss):
    def forward(self, logits, labels, **kwargs):
        shift_logits = logits[:, :-1, :].contiguous()
        shift_labels = labels[:, 1:].contiguous()
        return self._cross_entropy(shift_logits, shift_labels)


class _PrefLoss(SequenceLevelLoss):
    def forward(self, scores, targets, **kwargs):
        chosen, rejected = scores[:, 0], scores[:, 1]
        losses = -torch.log(torch.sigmoid(chosen - rejected) + 1e-8)
        return self._reduce(losses)


class _EM(BaseMetric):
    name = "em"

    def __init__(self):
        super().__init__()
        self._correct = self._total = 0

    def update(self, predictions, references, **kwargs):
        self._correct += sum(p == r for p, r in zip(predictions, references))
        self._total += len(predictions)

    def compute(self):
        return self._correct / self._total if self._total else 0.0

    def reset(self):
        super().reset()
        self._correct = self._total = 0


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestFullPipeline:
    def test_dataset_length(self):
        ds = ScenarioDataset(["A", "B", "C"], labels=["1", "0", "1"])
        assert len(ds) == 3

    def test_evaluator_full_report(self):
        ds = ScenarioDataset(
            ["Long scenario text here.", "Short one."],
            labels=["yes", "no"],
        )
        ev = DatasetEvaluator(ds)
        report = ev.evaluate()
        assert report["size"] == 2
        assert "label_distribution" in report

    def test_cot_monitor_detects(self):
        m = CoTMonitor(log_every=0)
        text = "Step 1: Identify. Step 2: Apply. Therefore: Done."
        traces = m.update([text])
        assert traces[0].has_cot

    def test_tot_monitor_detects(self):
        m = ToTMonitor(log_every=0)
        text = "Option 1: A. Option 2: B. Option 3: C. Best option: C."
        traces = m.update([text])
        assert traces[0].has_tot

    def test_aha_monitor_detects(self):
        m = AhaMonitor(log_every=0)
        text = "Wait, actually — I see now this is different."
        traces = m.update([text])
        assert traces[0].has_aha

    def test_metric_registry_pipeline(self):
        reg = MetricRegistry(metrics=[_EM()])
        reg.update_all(["A", "A", "B"], ["A", "B", "B"])
        results = reg.compute_all()
        assert results["em"] == pytest.approx(2 / 3, rel=1e-3)

    def test_token_loss_forward(self):
        loss_fn = _SimpleCELoss()
        logits = torch.randn(4, 16, 100)
        labels = torch.randint(0, 100, (4, 16))
        out = loss_fn(logits, labels)
        assert torch.isfinite(out)

    def test_sequence_loss_forward(self):
        loss_fn = _PrefLoss()
        scores = torch.tensor([[1.0, -1.0], [0.5, -0.5]])
        targets = torch.ones(2)
        out = loss_fn(scores, targets)
        assert torch.isfinite(out)

    def test_local_logger_writes_file(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            lg = LocalLogger(name="test", log_dir=tmpdir, use_console=False)
            lg.log_config({"lr": 1e-4})
            lg.log_step(0, {"loss": 2.5})
            lg.log_epoch(0, {"val_loss": 1.8})
            lg.close()
            files = os.listdir(tmpdir)
            assert any("metrics.jsonl" in f for f in files)

    def test_all_monitors_combined(self):
        texts = [
            "Step 1: think. Step 2: act. Therefore: done.",
            "Option 1: A. Option 2: B. Best option: A.",
            "Wait, actually — I see now it's different.",
            "The answer is 5.",
        ]
        cot = CoTMonitor(log_every=0)
        tot = ToTMonitor(log_every=0)
        aha = AhaMonitor(log_every=0)
        cot.update(texts)
        tot.update(texts)
        aha.update(texts)

        assert cot.get_stats()["total_samples"] == 4
        assert tot.get_stats()["total_samples"] == 4
        assert aha.get_stats()["total_samples"] == 4
