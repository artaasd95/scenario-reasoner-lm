"""
Base loss class hierarchy for Scenario Reasoner LM.

All custom loss functions used in scenario-based reasoning training should
inherit from :class:`BaseLoss`.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

import torch
import torch.nn as nn


class BaseLoss(nn.Module, ABC):
    """
    Abstract base class for scenario reasoning loss functions.

    All custom losses must inherit from this class and implement
    :meth:`forward`.  Inherits from :class:`torch.nn.Module` so instances
    integrate natively with PyTorch training loops.
    """

    def __init__(self, reduction: str = "mean") -> None:
        """
        Initialise the base loss.

        Args:
            reduction: Specifies the reduction to apply: ``"none"`` | ``"mean"`` | ``"sum"``.
        """
        super().__init__()
        if reduction not in ("none", "mean", "sum"):
            raise ValueError(
                f"Invalid reduction '{reduction}'. "
                "Choose from 'none', 'mean', or 'sum'."
            )
        self.reduction = reduction

    @abstractmethod
    def forward(self, *args: Any, **kwargs: Any) -> torch.Tensor:
        """
        Compute the loss.

        Returns:
            Loss tensor. Shape depends on ``self.reduction``.
        """

    def extra_repr(self) -> str:
        return f"reduction={self.reduction!r}"


class TokenLevelLoss(BaseLoss):
    """
    Base class for token-level losses (e.g. language modelling).

    Subclasses must override :meth:`forward`.  A cross-entropy helper is
    provided via :meth:`_cross_entropy`.

    Args:
        reduction: Reduction mode: ``"mean"`` | ``"sum"`` | ``"none"``.
        ignore_index: Token ID to ignore when computing the loss. Defaults to ``-100``.
    """

    def __init__(
        self,
        reduction: str = "mean",
        ignore_index: int = -100,
    ) -> None:
        super().__init__(reduction=reduction)
        self.ignore_index = ignore_index
        self._ce = nn.CrossEntropyLoss(
            reduction=reduction,
            ignore_index=ignore_index,
        )

    @abstractmethod
    def forward(
        self,
        logits: torch.Tensor,
        labels: torch.Tensor,
        **kwargs: Any,
    ) -> torch.Tensor:
        """
        Compute the token-level loss.

        Args:
            logits: Predicted logits of shape ``(batch, seq_len, vocab_size)``.
            labels: Target token IDs of shape ``(batch, seq_len)``.

        Returns:
            Loss tensor.
        """

    def _cross_entropy(
        self,
        logits: torch.Tensor,
        labels: torch.Tensor,
    ) -> torch.Tensor:
        """
        Apply standard cross-entropy over token logits.

        Args:
            logits: Shape ``(batch, seq_len, vocab_size)``.
            labels: Shape ``(batch, seq_len)``.

        Returns:
            Cross-entropy loss scalar (or per-token if reduction is ``"none"``).
        """
        batch_size, seq_len, vocab_size = logits.shape
        return self._ce(
            logits.view(batch_size * seq_len, vocab_size),
            labels.view(batch_size * seq_len),
        )

    def extra_repr(self) -> str:
        return (
            f"reduction={self.reduction!r}, "
            f"ignore_index={self.ignore_index}"
        )


class SequenceLevelLoss(BaseLoss):
    """
    Base class for sequence-level losses (e.g. reward shaping, RLHF).

    Subclasses must override :meth:`forward`.

    Args:
        reduction: Reduction mode: ``"mean"`` | ``"sum"`` | ``"none"``.
    """

    def __init__(self, reduction: str = "mean") -> None:
        super().__init__(reduction=reduction)

    @abstractmethod
    def forward(
        self,
        scores: torch.Tensor,
        targets: torch.Tensor,
        **kwargs: Any,
    ) -> torch.Tensor:
        """
        Compute the sequence-level loss.

        Args:
            scores: Per-sequence scalar scores, shape ``(batch,)``.
            targets: Target values / labels, shape ``(batch,)``.

        Returns:
            Loss tensor.
        """

    def _reduce(self, losses: torch.Tensor) -> torch.Tensor:
        """
        Apply the configured reduction to a per-sample loss vector.

        Args:
            losses: Per-sample losses, shape ``(batch,)``.

        Returns:
            Reduced scalar (or unchanged tensor if reduction is ``"none"``).
        """
        if self.reduction == "mean":
            return losses.mean()
        if self.reduction == "sum":
            return losses.sum()
        return losses
