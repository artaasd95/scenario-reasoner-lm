"""LLM provider configuration schema."""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field

ProviderKind = Literal["vllm", "ray_serve", "litellm", "ollama", "custom", "mock"]


class LLMConfig(BaseModel):
    """YAML-driven LLM provider configuration."""

    provider: ProviderKind = "mock"
    model_id: str = "mock"
    base_url: str | None = None
    api_key_env: str | None = Field(
        default=None,
        description="Env var name for BYOK API key (never commit the key itself).",
    )
    fallback_models: list[str] = Field(default_factory=list)
    extra: dict[str, Any] = Field(default_factory=dict)

    def resolve_api_key(self) -> str | None:
        if not self.api_key_env:
            return None
        import os

        return os.environ.get(self.api_key_env)
