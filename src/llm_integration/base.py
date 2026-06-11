"""LLM provider contracts for runtime inference (BYOK)."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any


@dataclass(slots=True)
class Completion:
    """Normalized LLM completion result."""

    text: str
    model_id: str
    backend_id: str
    token_usage: dict[str, int] = field(default_factory=dict)
    latency_ms: float = 0.0
    tool_calls: list[dict[str, Any]] | None = None
    metadata: dict[str, str] = field(default_factory=dict)


class LLMProvider(ABC):
    """Abstract provider for optional cloud/local LLM inference."""

    @property
    @abstractmethod
    def backend_id(self) -> str:
        """Stable adapter identifier (vllm, litellm, etc.)."""

    @abstractmethod
    async def complete(self, prompt: str, model_id: str, **kwargs: Any) -> Completion:
        """Return a completion for the given prompt and model."""
