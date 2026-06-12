"""Cloud LLM adapter with optional fallback routing (BYOK via env)."""

from __future__ import annotations

import time
from typing import Any

from src.llm_integration.adapters.mock_adapter import MockLLMAdapter
from src.llm_integration.base import Completion, LLMProvider
from src.llm_integration.config import LLMConfig
from src.llm_integration.context import resolve_max_tokens


class LiteLLMAdapter(LLMProvider):
    """LiteLLM-backed cloud provider; uses mock when no API key is set."""

    def __init__(self, config: LLMConfig) -> None:
        self._config = config
        self._mock = MockLLMAdapter(config)
        self._models = [config.model_id, *config.fallback_models]

    @property
    def backend_id(self) -> str:
        return "litellm"

    async def complete(self, prompt: str, model_id: str, **kwargs: Any) -> Completion:
        api_key = self._config.resolve_api_key()
        if not api_key:
            return await self._mock.complete(prompt, model_id, **kwargs)

        try:
            import litellm
        except ImportError:
            return await self._mock.complete(prompt, model_id, **kwargs)

        max_tokens = resolve_max_tokens(
            model_id,
            config=self._config,
            kwargs_max_tokens=kwargs.get("max_tokens"),
        )
        started = time.perf_counter()
        last_error: Exception | None = None
        for candidate in self._models or [model_id]:
            try:
                response = await litellm.acompletion(
                    model=candidate,
                    messages=[{"role": "user", "content": prompt}],
                    api_key=api_key,
                    max_tokens=max_tokens,
                )
                text = response.choices[0].message.content or ""
                usage = getattr(response, "usage", None) or {}
                latency_ms = (time.perf_counter() - started) * 1000
                return Completion(
                    text=str(text),
                    model_id=candidate,
                    backend_id=self.backend_id,
                    token_usage={
                        "prompt_tokens": int(getattr(usage, "prompt_tokens", 0) or 0),
                        "completion_tokens": int(getattr(usage, "completion_tokens", 0) or 0),
                    },
                    latency_ms=latency_ms,
                )
            except Exception as exc:  # noqa: BLE001 — try fallbacks
                last_error = exc
                continue

        if last_error:
            raise last_error
        return await self._mock.complete(prompt, model_id, **kwargs)
