"""HuggingFace dataset adapter (optional dependency)."""

from __future__ import annotations

from typing import Any, Dict, Iterator, List

from src.data.sources.base import DataSource


class HuggingFaceDataSource(DataSource):
    def __init__(self, dataset_name: str, split: str = "train", limit: int = 100) -> None:
        self.dataset_name = dataset_name
        self.split = split
        self.limit = limit

    def iter_rows(self) -> Iterator[Dict[str, Any]]:
        try:
            from datasets import load_dataset
        except ImportError as exc:
            raise ImportError("HuggingFaceDataSource requires `datasets` package") from exc

        ds = load_dataset(self.dataset_name, split=self.split)
        for i, row in enumerate(ds):
            if i >= self.limit:
                break
            yield {
                "path_id": f"hf-{self.dataset_name}-{i}",
                "input": row.get("input") or row.get("question", ""),
                "output": row.get("output") or row.get("answer", ""),
                "theta": dict(row.get("theta", {})),
            }
