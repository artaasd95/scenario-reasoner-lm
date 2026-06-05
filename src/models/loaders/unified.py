"""Unified model loaders — PyTorch, HF safetensors, hub/local (S10-05)."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional, Tuple


@dataclass
class LoadResult:
    model: Any
    tokenizer: Any
    source: str
    metadata: Dict[str, Any]


class UnifiedModelLoader:
    """
    Load models from local checkpoint, HF hub id, or safetensors index.

    Default path uses transformers when available; tests may inject stubs.
    """

    def __init__(self, config: Dict[str, Any]) -> None:
        self.config = config
        self.model_name_or_path = config.get("model_name_or_path", "")
        self.checkpoint = config.get("checkpoint")

    def load(self) -> LoadResult:
        checkpoint = self.checkpoint
        if checkpoint and Path(checkpoint).is_dir():
            return self._load_from_checkpoint(Path(checkpoint))
        if self.model_name_or_path:
            return self._load_from_hub(self.model_name_or_path)
        raise ValueError("UnifiedModelLoader requires checkpoint or model_name_or_path")

    def _load_from_hub(self, model_id: str) -> LoadResult:
        try:
            from transformers import AutoModelForCausalLM, AutoTokenizer
        except ImportError as exc:
            raise ImportError("transformers required for hub load") from exc

        tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
        model = AutoModelForCausalLM.from_pretrained(model_id, trust_remote_code=True)
        return LoadResult(model=model, tokenizer=tokenizer, source="hub", metadata={"model_id": model_id})

    def _load_from_checkpoint(self, path: Path) -> LoadResult:
        index = path / "model.safetensors.index.json"
        if index.is_file():
            meta = json.loads(index.read_text(encoding="utf-8"))
            return LoadResult(
                model=None,
                tokenizer=None,
                source="safetensors_index",
                metadata={"index": str(index), "weight_map_keys": len(meta.get("weight_map", {}))},
            )

        try:
            from transformers import AutoModelForCausalLM, AutoTokenizer
        except ImportError as exc:
            raise ImportError("transformers required for checkpoint load") from exc

        tokenizer = AutoTokenizer.from_pretrained(str(path), trust_remote_code=True)
        model = AutoModelForCausalLM.from_pretrained(str(path), trust_remote_code=True)
        return LoadResult(model=model, tokenizer=tokenizer, source="local", metadata={"path": str(path)})


def load_model(config: Dict[str, Any]) -> LoadResult:
    return UnifiedModelLoader(config).load()
