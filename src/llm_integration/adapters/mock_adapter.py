"""Default offline mock provider — no external services."""

from __future__ import annotations

import time
from typing import Any

from src.llm_integration.base import Completion, LLMProvider
from src.llm_integration.config import LLMConfig


class MockLLMAdapter(LLMProvider):
    """Deterministic mock completions for tests and offline defaults."""

    def __init__(self, config: LLMConfig) -> None:
        self._config = config

    @property
    def backend_id(self) -> str:
        return "mock"

    async def complete(self, prompt: str, model_id: str, **kwargs: Any) -> Completion:
        started = time.perf_counter()
        text = f"[mock:{model_id}] processed {len(prompt)} chars"
        latency_ms = (time.perf_counter() - started) * 1000
        return Completion(
            text=text,
            model_id=model_id,
            backend_id=self.backend_id,
            token_usage={"prompt_tokens": len(prompt.split()), "completion_tokens": 8},
            latency_ms=latency_ms,
            metadata={"mode": "offline"},
        )
