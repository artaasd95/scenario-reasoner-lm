"""Unified model loader tests (S10-05)."""

from __future__ import annotations

import json
from types import SimpleNamespace

from src.models.loaders.unified import UnifiedModelLoader


class TestModelLoaders:
    def test_checkpoint_load_with_safetensors_index(self, tmp_path, monkeypatch):
        index = {"weight_map": {"layer.0": "model.safetensors"}}
        (tmp_path / "model.safetensors.index.json").write_text(json.dumps(index))
        (tmp_path / "config.json").write_text(json.dumps({"model_type": "gpt2"}))

        class _Tok:
            pad_token = None
            eos_token = "<eos>"

        monkeypatch.setattr(
            "transformers.AutoTokenizer.from_pretrained",
            lambda *a, **k: _Tok(),
        )
        monkeypatch.setattr(
            "transformers.AutoModelForCausalLM.from_pretrained",
            lambda *a, **k: SimpleNamespace(),
        )

        loader = UnifiedModelLoader({"checkpoint": str(tmp_path)})
        result = loader.load()
        assert result.source == "local"
        assert result.model is not None
        assert result.metadata["weight_map_keys"] == 1

    def test_hub_load_mock(self, monkeypatch):
        class _Tok:
            pad_token = "<eos>"
            eos_token = "<eos>"

        monkeypatch.setattr(
            "transformers.AutoTokenizer.from_pretrained",
            lambda *a, **k: _Tok(),
        )
        monkeypatch.setattr(
            "transformers.AutoModelForCausalLM.from_pretrained",
            lambda *a, **k: SimpleNamespace(),
        )
        result = UnifiedModelLoader({"model_name_or_path": "gpt2"}).load()
        assert result.source == "hub"
        assert result.tokenizer is not None
