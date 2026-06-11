"""Ray Serve LLM adapter with mock fallback when cluster unavailable."""

from __future__ import annotations

from typing import Any

from src.llm_integration.adapters._http_base import OpenAICompatibleAdapter
from src.llm_integration.adapters.mock_adapter import MockLLMAdapter
from src.llm_integration.base import Completion
from src.llm_integration.config import LLMConfig


class RayServeLLMAdapter(OpenAICompatibleAdapter):
    """Multi-replica Ray Serve endpoint; falls back to mock in tests."""

    def __init__(self, config: LLMConfig) -> None:
        base_url = config.base_url or "http://localhost:8000"
        super().__init__(config, backend_id="ray_serve", default_base_url=base_url)
        self._mock = MockLLMAdapter(config)
        self._use_mock = bool(config.extra.get("mock_without_cluster", False))

    async def complete(self, prompt: str, model_id: str, **kwargs: Any) -> Completion:
        if self._use_mock:
            completion = await self._mock.complete(prompt, model_id, **kwargs)
            return completion.model_copy(
                update={"backend_id": self.backend_id, "metadata": {"mock_without_cluster": "true"}}
            )
        return await super().complete(prompt, model_id, **kwargs)
