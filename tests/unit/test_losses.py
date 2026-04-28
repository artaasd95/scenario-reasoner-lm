"""
Unit tests for base loss classes.
"""

import pytest
import torch
from typing import Any

from src.losses.base_losses import BaseLoss, TokenLevelLoss, SequenceLevelLoss


# ---------------------------------------------------------------------------
# Concrete subclasses for testing
# ---------------------------------------------------------------------------

class _ConcreteTokenLoss(TokenLevelLoss):
    """Simple next-token prediction loss."""

    def forward(self, logits: torch.Tensor, labels: torch.Tensor, **kwargs: Any) -> torch.Tensor:
        shift_logits = logits[:, :-1, :].contiguous()
        shift_labels = labels[:, 1:].contiguous()
        return self._cross_entropy(shift_logits, shift_labels)


class _ConcreteSeqLoss(SequenceLevelLoss):
    """Simple log-sigmoid preference loss."""

    def forward(self, scores: torch.Tensor, targets: torch.Tensor, **kwargs: Any) -> torch.Tensor:
        chosen, rejected = scores[:, 0], scores[:, 1]
        losses = -torch.log(torch.sigmoid(chosen - rejected) + 1e-8)
        return self._reduce(losses)


# ---------------------------------------------------------------------------
# BaseLoss tests
# ---------------------------------------------------------------------------

class TestBaseLoss:
    def test_abstract_cannot_instantiate(self):
        with pytest.raises(TypeError):
            BaseLoss()  # type: ignore[abstract]

    def test_invalid_reduction_raises(self):
        with pytest.raises(ValueError):
            _ConcreteTokenLoss(reduction="bad")

    def test_is_nn_module(self):
        import torch.nn as nn
        loss = _ConcreteTokenLoss()
        assert isinstance(loss, nn.Module)


# ---------------------------------------------------------------------------
# TokenLevelLoss tests
# ---------------------------------------------------------------------------

class TestTokenLevelLoss:
    def _make_batch(self, batch=4, seq=16, vocab=100):
        logits = torch.randn(batch, seq, vocab)
        labels = torch.randint(0, vocab, (batch, seq))
        return logits, labels

    def test_forward_returns_scalar(self):
        loss_fn = _ConcreteTokenLoss()
        logits, labels = self._make_batch()
        out = loss_fn(logits, labels)
        assert out.ndim == 0

    def test_forward_with_ignore_index(self):
        loss_fn = _ConcreteTokenLoss(ignore_index=0)
        logits, labels = self._make_batch()
        labels[:, 0] = 0  # mark some positions as ignored
        out = loss_fn(logits, labels)
        assert torch.isfinite(out)

    def test_reduction_sum(self):
        loss_mean = _ConcreteTokenLoss(reduction="mean")
        loss_sum = _ConcreteTokenLoss(reduction="sum")
        logits, labels = self._make_batch(batch=4, seq=8, vocab=50)
        out_mean = loss_mean(logits, labels)
        out_sum = loss_sum(logits, labels)
        assert out_sum.item() > out_mean.item()

    def test_extra_repr(self):
        loss_fn = _ConcreteTokenLoss(reduction="sum", ignore_index=-100)
        r = loss_fn.extra_repr()
        assert "sum" in r and "-100" in r


# ---------------------------------------------------------------------------
# SequenceLevelLoss tests
# ---------------------------------------------------------------------------

class TestSequenceLevelLoss:
    def test_forward_returns_scalar(self):
        loss_fn = _ConcreteSeqLoss()
        scores = torch.tensor([[1.0, -1.0], [0.5, -0.5]])
        targets = torch.ones(2)
        out = loss_fn(scores, targets)
        assert out.ndim == 0

    def test_reduction_none(self):
        loss_fn = _ConcreteSeqLoss(reduction="none")
        scores = torch.tensor([[1.0, -1.0], [0.5, -0.5]])
        targets = torch.ones(2)
        out = loss_fn(scores, targets)
        assert out.shape == (2,)

    def test_reduction_sum(self):
        loss_mean = _ConcreteSeqLoss(reduction="mean")
        loss_sum = _ConcreteSeqLoss(reduction="sum")
        scores = torch.tensor([[1.0, -1.0], [0.5, -0.5]])
        targets = torch.ones(2)
        assert loss_sum(scores, targets).item() == pytest.approx(
            loss_mean(scores, targets).item() * 2, rel=1e-5
        )
