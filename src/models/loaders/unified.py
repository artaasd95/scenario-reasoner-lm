"""Unified model loaders — PyTorch, HF safetensors, hub/local, PEFT adapters (S10-05)."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional


@dataclass
class LoadResult:
    model: Any
    tokenizer: Any
    source: str
    metadata: Dict[str, Any]


class UnifiedModelLoader:
    """
    Load models from local checkpoint, HF hub id, safetensors index, or base+adapter.

    Config keys:
        model_name_or_path: Hub id or local base model path
        checkpoint: Local full checkpoint directory
        adapter_path: LoRA adapter directory (loads PeftModel on base)
    """

    def __init__(self, config: Dict[str, Any]) -> None:
        self.config = config
        self.model_name_or_path = config.get("model_name_or_path", "")
        self.checkpoint = config.get("checkpoint")
        self.adapter_path = config.get("adapter_path")

    def load(self) -> LoadResult:
        if self.adapter_path:
            return self._load_with_adapter(Path(self.adapter_path))
        checkpoint = self.checkpoint
        if checkpoint and Path(checkpoint).is_dir():
            return self._load_from_checkpoint(Path(checkpoint))
        if self.model_name_or_path:
            return self._load_from_hub(self.model_name_or_path)
        raise ValueError(
            "UnifiedModelLoader requires adapter_path, checkpoint, or model_name_or_path"
        )

    def _load_with_adapter(self, adapter_path: Path) -> LoadResult:
        if not adapter_path.is_dir():
            raise FileNotFoundError(f"adapter_path not found: {adapter_path}")

        adapter_config = adapter_path / "adapter_config.json"
        if adapter_config.is_file():
            meta = json.loads(adapter_config.read_text(encoding="utf-8"))
            base_name = self.model_name_or_path or meta.get("base_model_name_or_path", "")
            if not base_name:
                raise ValueError(
                    "model_name_or_path required when loading adapter (or adapter_config base)"
                )
            try:
                import torch
                from peft import PeftModel
                from transformers import AutoModelForCausalLM, AutoTokenizer
            except ImportError as exc:
                raise ImportError("transformers and peft required for adapter load") from exc

            tokenizer = AutoTokenizer.from_pretrained(str(adapter_path), trust_remote_code=True)
            if tokenizer.pad_token is None:
                tokenizer.pad_token = tokenizer.eos_token
            base_model = AutoModelForCausalLM.from_pretrained(
                base_name,
                torch_dtype=torch.bfloat16,
                device_map="auto",
                trust_remote_code=True,
            )
            model = PeftModel.from_pretrained(base_model, str(adapter_path))
            return LoadResult(
                model=model,
                tokenizer=tokenizer,
                source="peft_adapter",
                metadata={
                    "adapter_path": str(adapter_path),
                    "base_model": base_name,
                    "peft_type": meta.get("peft_type"),
                },
            )

        return self._load_from_checkpoint(adapter_path)

    def _load_from_hub(self, model_id: str) -> LoadResult:
        try:
            from transformers import AutoModelForCausalLM, AutoTokenizer
        except ImportError as exc:
            raise ImportError("transformers required for hub load") from exc

        tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
        model = AutoModelForCausalLM.from_pretrained(model_id, trust_remote_code=True)
        return LoadResult(
            model=model,
            tokenizer=tokenizer,
            source="hub",
            metadata={"model_id": model_id},
        )

    def _load_from_checkpoint(self, path: Path) -> LoadResult:
        try:
            from transformers import AutoModelForCausalLM, AutoTokenizer
        except ImportError as exc:
            raise ImportError("transformers required for checkpoint load") from exc

        index = path / "model.safetensors.index.json"
        tokenizer = AutoTokenizer.from_pretrained(str(path), trust_remote_code=True)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        model = AutoModelForCausalLM.from_pretrained(str(path), trust_remote_code=True)
        metadata: Dict[str, Any] = {"path": str(path)}
        if index.is_file():
            meta = json.loads(index.read_text(encoding="utf-8"))
            metadata["safetensors_index"] = str(index)
            metadata["weight_map_keys"] = len(meta.get("weight_map", {}))
        return LoadResult(
            model=model,
            tokenizer=tokenizer,
            source="local",
            metadata=metadata,
        )


def load_model(config: Dict[str, Any]) -> LoadResult:
    return UnifiedModelLoader(config).load()
