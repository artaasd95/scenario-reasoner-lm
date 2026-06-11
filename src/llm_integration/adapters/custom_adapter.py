"""BYOK template for custom internal LLM services."""

from __future__ import annotations

from typing import Any

from src.llm_integration.adapters._http_base import OpenAICompatibleAdapter
from src.llm_integration.adapters.litellm_adapter import LiteLLMAdapter
from src.llm_integration.adapters.mock_adapter import MockLLMAdapter
from src.llm_integration.base import Completion
from src.llm_integration.config import LLMConfig


class CustomAdapter(OpenAICompatibleAdapter):
    """
    Extension point for org-specific inference endpoints.

    Configure `base_url` and `api_key_env` in configs/llm_custom.yaml.
    Set `extra.fallback_provider: litellm` to chain to a cloud fallback.
    """

    def __init__(self, config: LLMConfig) -> None:
        base_url = config.base_url or "http://localhost:9000"
        super().__init__(config, backend_id="custom", default_base_url=base_url)
        self._fallback_kind = str(config.extra.get("fallback_provider", "")).lower()
        self._mock = MockLLMAdapter(config)
        self._litellm = LiteLLMAdapter(config) if self._fallback_kind == "litellm" else None

    async def complete(self, prompt: str, model_id: str, **kwargs: Any) -> Completion:
        if not self._config.base_url:
            return await self._mock.complete(prompt, model_id, **kwargs)
        try:
            return await super().complete(prompt, model_id, **kwargs)
        except Exception:  # noqa: BLE001 — optional fallback chain
            if self._litellm is not None:
                return await self._litellm.complete(prompt, model_id, **kwargs)
            return await self._mock.complete(prompt, model_id, **kwargs)
