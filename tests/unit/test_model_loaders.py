"""Unified model loader tests (S10-05)."""

from __future__ import annotations

import json

from src.models.loaders.unified import UnifiedModelLoader


class TestModelLoaders:
    def test_safetensors_index_metadata(self, tmp_path):
        index = {"weight_map": {"layer.0": "model.safetensors"}}
        (tmp_path / "model.safetensors.index.json").write_text(json.dumps(index))
        loader = UnifiedModelLoader({"checkpoint": str(tmp_path)})
        result = loader.load()
        assert result.source == "safetensors_index"
        assert result.metadata["weight_map_keys"] == 1
