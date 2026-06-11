"""Shared HTTP helper for OpenAI-compatible adapters."""

from __future__ import annotations

import time
from typing import Any

import httpx

from src.llm_integration.base import Completion, LLMProvider
from src.llm_integration.config import LLMConfig


class OpenAICompatibleAdapter(LLMProvider):
    """Base for vLLM/Ollama/custom OpenAI-compatible endpoints."""

    def __init__(self, config: LLMConfig, *, backend_id: str, default_base_url: str) -> None:
        self._config = config
        self._backend_id = backend_id
        self._base_url = (config.base_url or default_base_url).rstrip("/")

    @property
    def backend_id(self) -> str:
        return self._backend_id

    async def complete(self, prompt: str, model_id: str, **kwargs: Any) -> Completion:
        api_key = self._config.resolve_api_key()
        headers: dict[str, str] = {"Content-Type": "application/json"}
        if api_key:
            headers["Authorization"] = f"Bearer {api_key}"

        payload = {
            "model": model_id,
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": int(kwargs.get("max_tokens", 256)),
        }
        url = f"{self._base_url}/v1/chat/completions"
        started = time.perf_counter()
        async with httpx.AsyncClient(timeout=60.0) as client:
            response = await client.post(url, json=payload, headers=headers)
            response.raise_for_status()
            data = response.json()

        choice = data["choices"][0]["message"]["content"]
        usage = data.get("usage", {})
        latency_ms = (time.perf_counter() - started) * 1000
        return Completion(
            text=str(choice),
            model_id=model_id,
            backend_id=self._backend_id,
            token_usage={
                "prompt_tokens": int(usage.get("prompt_tokens", 0)),
                "completion_tokens": int(usage.get("completion_tokens", 0)),
            },
            latency_ms=latency_ms,
        )
