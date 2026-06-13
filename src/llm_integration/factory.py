"""Factory for YAML-driven LLM provider selection."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

from src.llm_integration.adapters import (
    CustomAdapter,
    LiteLLMAdapter,
    MockLLMAdapter,
    OllamaAdapter,
    RayServeLLMAdapter,
    VLLMAdapter,
)
from src.llm_integration.base import LLMProvider
from src.llm_integration.config import LLMConfig

_REGISTRY: dict[str, type[LLMProvider]] = {
    "mock": MockLLMAdapter,
    "vllm": VLLMAdapter,
    "ray_serve": RayServeLLMAdapter,
    "litellm": LiteLLMAdapter,
    "ollama": OllamaAdapter,
    "custom": CustomAdapter,
}

_PROVIDER_CONFIG_MAP = {
    "mock": "configs/llm_mock.yaml",
    "stub": "configs/llm_mock.yaml",
    "ollama": "configs/llm_ollama.yaml",
    "vllm": "configs/llm_single_gpu.yaml",
    "ray_serve": "configs/llm_distributed.yaml",
    "litellm": "configs/llm_cloud.yaml",
    "custom": "configs/llm_custom.yaml",
}


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def create_llm_provider_for_name(provider: str) -> LLMProvider:
    """Resolve provider alias or config path to an LLMProvider instance."""
    path = Path(provider)
    if path.exists():
        return create_llm_provider(path)
    mapped = _PROVIDER_CONFIG_MAP.get(provider.lower())
    if mapped is None:
        raise ValueError(f"Unknown LLM provider alias: {provider!r}")
    return create_llm_provider(_repo_root() / mapped)


def create_llm_provider(config_path: str | Path) -> LLMProvider:
    """Load YAML config and return the matching provider."""
    path = Path(config_path)
    if not path.exists():
        raise FileNotFoundError(f"LLM config not found: {path}")

    raw: dict[str, Any] = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    config = LLMConfig.model_validate(raw)
    provider_cls = _REGISTRY.get(config.provider)
    if provider_cls is None:
        known = ", ".join(sorted(_REGISTRY))
        raise ValueError(f"Unknown LLM provider {config.provider!r}; expected one of: {known}")
    return provider_cls(config)
