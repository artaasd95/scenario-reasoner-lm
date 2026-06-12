"""Optional LLM provider layer — BYOK for runtime inference only."""

from src.llm_integration.base import Completion, LLMProvider
from src.llm_integration.config import LLMConfig
from src.llm_integration.context import (
    AssembledContext,
    ContextBudget,
    ContextSegment,
    assemble_context,
    estimate_tokens,
    resolve_max_tokens,
    resolve_model_limits,
)
from src.llm_integration.factory import create_llm_provider

__all__ = [
    "AssembledContext",
    "Completion",
    "ContextBudget",
    "ContextSegment",
    "LLMConfig",
    "LLMProvider",
    "assemble_context",
    "create_llm_provider",
    "estimate_tokens",
    "resolve_max_tokens",
    "resolve_model_limits",
]
