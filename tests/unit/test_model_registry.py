"""Unit tests for Qwen portfolio model registry."""

from __future__ import annotations

import os
from pathlib import Path

import pytest

from src.models.model_registry import ModelRegistry


@pytest.fixture
def registry():
    return ModelRegistry.load()


def test_get_known_model(registry):
    entry = registry.get("qwen3-0.6b")
    assert entry.hub_id == "Qwen/Qwen3-0.6B"
    assert entry.tier == "smoke"


def test_get_by_tier_smoke(registry):
    ids = [e.model_id for e in registry.get_by_tier("smoke")]
    assert "qwen3-0.6b" in ids
    assert "qwen2.5-0.5b" in ids


def test_require_models_root_missing(monkeypatch):
    monkeypatch.delenv("SCENARIO_MODELS_ROOT", raising=False)
    with pytest.raises(EnvironmentError, match="SCENARIO_MODELS_ROOT"):
        ModelRegistry.require_models_root()


def test_resolve_path_with_env(registry, monkeypatch, tmp_path):
    monkeypatch.setenv("SCENARIO_MODELS_ROOT", str(tmp_path))
    local = tmp_path / "Qwen" / "Qwen3-0.6B"
    local.mkdir(parents=True)
    resolved = registry.resolve_path("qwen3-0.6b")
    assert resolved == local


def test_to_train_config(registry, monkeypatch, tmp_path):
    monkeypatch.setenv("SCENARIO_MODELS_ROOT", str(tmp_path))
    local = tmp_path / "Qwen" / "Qwen3-0.6B"
    local.mkdir(parents=True)
    cfg = registry.to_train_config("qwen3-0.6b")
    assert cfg["model_id"] == "qwen3-0.6b"
    assert cfg["model_name_or_path"] == str(local)
    assert cfg["hub_id"] == "Qwen/Qwen3-0.6B"
