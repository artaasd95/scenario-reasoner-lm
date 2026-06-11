from __future__ import annotations

import pytest

from src.llm_integration.adapters.mock_adapter import MockLLMAdapter
from src.llm_integration.base import LLMProvider
from src.llm_integration.config import LLMConfig


@pytest.mark.asyncio
async def test_mock_provider() -> None:
    provider = MockLLMAdapter(LLMConfig(provider="mock", model_id="mock"))
    assert isinstance(provider, LLMProvider)
    out = await provider.complete("hello", "mock")
    assert out.backend_id == "mock"
