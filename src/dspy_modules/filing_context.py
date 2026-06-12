"""Budget-aware assembly of 10-K filing excerpts for DSPy modules."""

from __future__ import annotations

import os
from typing import List

from src.llm_integration.context import (
    AssembledContext,
    ContextBudget,
    ContextSegment,
    PRIORITY_CRITICAL,
    PRIORITY_RETRIEVED,
    max_context_from_env,
    resolve_model_limits,
)
from src.risk.schema import EvidenceChunk

SECTION_RANK = {
    "Risk Factors": 0,
    "MD&A": 1,
    "Legal Proceedings": 2,
    "Cybersecurity": 3,
    "Regulatory": 4,
    "Supply Chain": 5,
    "Business": 6,
}

DEFAULT_EXCERPT_INPUT_TOKENS = 4096


def _section_priority(section_name: str) -> int:
    return SECTION_RANK.get(section_name, 99)


def build_filing_excerpt(
    chunks: List[EvidenceChunk],
    *,
    model_id: str | None = None,
    max_input_tokens: int | None = None,
) -> AssembledContext:
    """
    Pack filing chunks into a token-budgeted excerpt.

    Chunks are ranked by section relevance (Risk Factors first), then chunk_id.
    """
    resolved_model = model_id or os.getenv("ENTERPRISE_MODEL_NAME", "gpt-4o-mini")
    if max_input_tokens is None:
        env_cap = max_context_from_env()
        if env_cap is not None:
            max_input_tokens = resolve_model_limits(
                resolved_model,
                max_context_override=env_cap,
            ).max_input_tokens
        else:
            max_input_tokens = DEFAULT_EXCERPT_INPUT_TOKENS

    ordered = sorted(
        chunks,
        key=lambda c: (_section_priority(c.section_name), c.chunk_id),
    )
    segments: list[ContextSegment] = []
    for chunk in ordered:
        rank = _section_priority(chunk.section_name)
        segments.append(
            ContextSegment(
                name=chunk.chunk_id,
                content=f"[{chunk.chunk_id}] ({chunk.section_name})\n{chunk.quote_text}",
                priority=PRIORITY_CRITICAL if rank <= 1 else PRIORITY_RETRIEVED,
                protected=rank == 0,
            )
        )

    return ContextBudget(
        resolved_model,
        max_input_tokens=max_input_tokens,
    ).assemble(segments)
