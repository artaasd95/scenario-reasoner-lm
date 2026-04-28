"""
HuggingFace dataset and model wrappers for Scenario Reasoner LM.

Wraps ``datasets.Dataset`` objects and HuggingFace tokenisers/models into
PyTorch-compatible form that integrates with the rest of this project.
"""

from __future__ import annotations

from typing import Any, Callable, Dict, List, Optional

import torch

from src.data.base_dataset import BaseScenarioDataset


class HFDatasetWrapper(BaseScenarioDataset):
    """
    Wraps a HuggingFace ``datasets.Dataset`` as a PyTorch-compatible dataset.

    The wrapper lazily applies an optional ``transform`` callable to each
    sample, making it straightforward to plug in a tokeniser or any other
    per-example processing function.

    Example::

        from datasets import load_dataset
        from src.data.hf_datasets import HFDatasetWrapper

        raw = load_dataset("squad", split="train")
        dataset = HFDatasetWrapper(raw, split="train")

    Args:
        hf_dataset: A loaded ``datasets.Dataset`` instance.
        split: Dataset split label (e.g. ``"train"``).
        transform: Optional callable applied to each sample dictionary
            before it is returned by :meth:`__getitem__`.
        metadata: Optional dataset-level metadata dictionary.
    """

    def __init__(
        self,
        hf_dataset: Any,
        split: str = "train",
        transform: Optional[Callable[[Dict[str, Any]], Dict[str, Any]]] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> None:
        super().__init__(split=split, metadata=metadata)
        self._hf_dataset = hf_dataset
        self.transform = transform

    def __len__(self) -> int:
        return len(self._hf_dataset)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        sample: Dict[str, Any] = dict(self._hf_dataset[idx])
        sample = self.preprocess(sample)
        if self.transform is not None:
            sample = self.transform(sample)
        return sample

    @property
    def column_names(self) -> List[str]:
        """Return the column names of the underlying HF dataset."""
        return list(self._hf_dataset.column_names)

    @property
    def features(self) -> Any:
        """Return the HuggingFace ``Features`` schema."""
        return self._hf_dataset.features

    def filter(self, function: Callable, **kwargs: Any) -> "HFDatasetWrapper":
        """
        Apply a HuggingFace ``filter`` operation and return a new wrapper.

        Args:
            function: Filter predicate.
            **kwargs: Additional keyword arguments forwarded to ``Dataset.filter``.

        Returns:
            A new :class:`HFDatasetWrapper` over the filtered dataset.
        """
        filtered = self._hf_dataset.filter(function, **kwargs)
        return HFDatasetWrapper(
            filtered,
            split=self.split,
            transform=self.transform,
            metadata=self.metadata,
        )

    def map(self, function: Callable, **kwargs: Any) -> "HFDatasetWrapper":
        """
        Apply a HuggingFace ``map`` operation and return a new wrapper.

        Args:
            function: Mapping function.
            **kwargs: Additional keyword arguments forwarded to ``Dataset.map``.

        Returns:
            A new :class:`HFDatasetWrapper` over the mapped dataset.
        """
        mapped = self._hf_dataset.map(function, **kwargs)
        return HFDatasetWrapper(
            mapped,
            split=self.split,
            transform=self.transform,
            metadata=self.metadata,
        )


class HFTokenizedDataset(HFDatasetWrapper):
    """
    A HuggingFace dataset wrapper that automatically tokenises text fields.

    Tokenisation is performed lazily per-sample in :meth:`__getitem__`.

    Example::

        from datasets import load_dataset
        from transformers import AutoTokenizer
        from src.data.hf_datasets import HFTokenizedDataset

        tokenizer = AutoTokenizer.from_pretrained("gpt2")
        raw = load_dataset("squad", split="train")
        dataset = HFTokenizedDataset(
            raw,
            tokenizer=tokenizer,
            text_field="question",
            max_length=256,
        )

    Args:
        hf_dataset: A loaded ``datasets.Dataset`` instance.
        tokenizer: A HuggingFace tokeniser instance.
        text_field: Name of the column containing the text to tokenise.
        max_length: Maximum token sequence length (truncation/padding applied).
        split: Dataset split label.
        metadata: Optional dataset-level metadata dictionary.
    """

    def __init__(
        self,
        hf_dataset: Any,
        tokenizer: Any,
        text_field: str = "text",
        max_length: int = 512,
        split: str = "train",
        metadata: Optional[Dict[str, Any]] = None,
    ) -> None:
        self.tokenizer = tokenizer
        self.text_field = text_field
        self.max_length = max_length

        def _tokenize(sample: Dict[str, Any]) -> Dict[str, Any]:
            encoded = tokenizer(
                sample[text_field],
                max_length=max_length,
                padding="max_length",
                truncation=True,
                return_tensors="pt",
            )
            return {
                **sample,
                "input_ids": encoded["input_ids"].squeeze(0),
                "attention_mask": encoded["attention_mask"].squeeze(0),
            }

        super().__init__(
            hf_dataset,
            split=split,
            transform=_tokenize,
            metadata=metadata,
        )

    def collate_fn(
        self, batch: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """Collate tokenised samples, stacking tensor fields."""
        collated: Dict[str, Any] = {}
        tensor_keys = {"input_ids", "attention_mask"}
        for key in batch[0]:
            values = [sample[key] for sample in batch]
            if key in tensor_keys and isinstance(values[0], torch.Tensor):
                collated[key] = torch.stack(values)
            else:
                collated[key] = values
        return collated
