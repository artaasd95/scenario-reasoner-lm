"""
Abstract base dataset for Scenario Reasoner LM.

All dataset implementations — whether backed by HuggingFace or custom data
sources — must inherit from BaseScenarioDataset.
"""

from abc import ABC, abstractmethod
from typing import Any, Dict, Iterator, List, Optional

import torch
from torch.utils.data import Dataset


class BaseScenarioDataset(Dataset, ABC):
    """
    Abstract base class for scenario-based reasoning datasets.

    Subclasses must implement :meth:`__len__` and :meth:`__getitem__`.
    Optional hooks (:meth:`preprocess`, :meth:`collate_fn`) can be
    overridden to customise batching and preprocessing behaviour.

    Attributes:
        split: Dataset split identifier (e.g. ``"train"``, ``"val"``, ``"test"``).
        metadata: Arbitrary key-value metadata for this dataset instance.
    """

    def __init__(
        self,
        split: str = "train",
        metadata: Optional[Dict[str, Any]] = None,
    ) -> None:
        """
        Initialise the base dataset.

        Args:
            split: Dataset split (``"train"``, ``"val"``, or ``"test"``).
            metadata: Optional dictionary with dataset-level metadata.
        """
        super().__init__()
        self.split = split
        self.metadata: Dict[str, Any] = metadata or {}

    # ------------------------------------------------------------------
    # Required interface
    # ------------------------------------------------------------------

    @abstractmethod
    def __len__(self) -> int:
        """Return the total number of samples."""

    @abstractmethod
    def __getitem__(self, idx: int) -> Dict[str, Any]:
        """
        Return a single sample.

        Args:
            idx: Zero-based sample index.

        Returns:
            A dictionary containing at minimum the keys ``"input"`` and
            ``"label"``.  Subclasses may add additional keys (e.g.
            ``"reasoning_trace"``, ``"scenario_id"``).
        """

    # ------------------------------------------------------------------
    # Optional hooks
    # ------------------------------------------------------------------

    def preprocess(self, sample: Dict[str, Any]) -> Dict[str, Any]:
        """
        Apply preprocessing to a raw sample.

        Override this method to add tokenisation, normalisation, or any
        other per-sample transformation.  The default implementation
        returns the sample unchanged.

        Args:
            sample: Raw sample dictionary.

        Returns:
            Preprocessed sample dictionary.
        """
        return sample

    def collate_fn(
        self, batch: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """
        Collate a list of samples into a batch.

        The default implementation stacks torch tensors and keeps other
        values as lists.  Override for custom batching logic.

        Args:
            batch: List of sample dictionaries from :meth:`__getitem__`.

        Returns:
            Batched dictionary suitable for model input.
        """
        collated: Dict[str, Any] = {}
        for key in batch[0]:
            values = [sample[key] for sample in batch]
            if isinstance(values[0], torch.Tensor):
                collated[key] = torch.stack(values)
            else:
                collated[key] = values
        return collated

    def get_dataloader(
        self,
        batch_size: int = 16,
        shuffle: Optional[bool] = None,
        num_workers: int = 0,
        pin_memory: bool = False,
    ) -> "torch.utils.data.DataLoader":
        """
        Create a :class:`~torch.utils.data.DataLoader` for this dataset.

        Args:
            batch_size: Number of samples per batch.
            shuffle: Whether to shuffle.  Defaults to ``True`` for the
                ``"train"`` split and ``False`` otherwise.
            num_workers: Number of worker processes for data loading.
            pin_memory: Pin tensors to CUDA page-locked memory.

        Returns:
            Configured :class:`~torch.utils.data.DataLoader`.
        """
        from torch.utils.data import DataLoader

        if shuffle is None:
            shuffle = self.split == "train"

        return DataLoader(
            self,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=num_workers,
            pin_memory=pin_memory,
            collate_fn=self.collate_fn,
        )

    def __iter__(self) -> Iterator[Dict[str, Any]]:
        for i in range(len(self)):
            yield self[i]

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}("
            f"split={self.split!r}, "
            f"size={len(self)})"
        )
