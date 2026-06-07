"""Distilled teacher traces — DPO pairs from Colab JSONL (S13-04)."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List

from src.data.sources.file_source import FileDataSource


class DistilledDataSource:
    """Load DPO preference pairs from a distilled manifest."""

    def __init__(self, manifest_path: str | Path) -> None:
        self.manifest_path = Path(manifest_path)
        with open(self.manifest_path, encoding="utf-8") as f:
            self.manifest: Dict[str, Any] = json.load(f)
        self.root = self.manifest_path.parent

    @property
    def dpo_path(self) -> Path:
        filename = self.manifest.get("dpo_pairs_file", "dpo_pairs.jsonl")
        return self.root / filename

    def load_dpo_pairs(self) -> List[Dict[str, str]]:
        pairs: List[Dict[str, str]] = []
        for row in FileDataSource(self.dpo_path).iter_rows():
            pairs.append({
                "prompt": row["prompt"],
                "chosen": row["chosen"],
                "rejected": row["rejected"],
            })
        return pairs

    def load_sft_rows(self) -> List[Dict[str, Any]]:
        filename = self.manifest.get("sft_file", "sft.jsonl")
        path = self.root / filename
        if not path.is_file():
            return []
        return list(FileDataSource(path).iter_rows())
