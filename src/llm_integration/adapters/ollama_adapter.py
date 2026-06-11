"""Ollama local inference adapter."""

from __future__ import annotations

from src.llm_integration.adapters._http_base import OpenAICompatibleAdapter
from src.llm_integration.config import LLMConfig


class OllamaAdapter(OpenAICompatibleAdapter):
    def __init__(self, config: LLMConfig) -> None:
        super().__init__(config, backend_id="ollama", default_base_url="http://localhost:11434")
