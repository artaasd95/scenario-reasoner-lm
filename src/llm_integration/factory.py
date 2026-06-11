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
