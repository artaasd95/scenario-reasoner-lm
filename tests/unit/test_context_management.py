from __future__ import annotations

from src.dspy_modules.filing_context import build_filing_excerpt
from src.llm_integration.context import ContextBudget, resolve_model_limits
from src.risk.schema import EvidenceChunk


def _chunk(chunk_id: str, section: str, text: str) -> EvidenceChunk:
    return EvidenceChunk(
        section_name=section,
        chunk_id=chunk_id,
        source_span="0:10",
        quote_text=text,
    )


def test_build_filing_excerpt_keeps_risk_factors_first() -> None:
    chunks = [
        _chunk("c2", "Business", "business " * 100),
        _chunk("c1", "Risk Factors", "risk " * 50),
        _chunk("c3", "MD&A", "mdna " * 100),
    ]
    assembled = build_filing_excerpt(chunks, max_input_tokens=120)
    assert "c1" in assembled.segments_kept
    assert assembled.estimated_tokens <= assembled.max_input_tokens


def test_build_filing_excerpt_drops_overflow_chunks() -> None:
    chunks = [
        _chunk(f"c{i}", "Business", "x " * 500) for i in range(10)
    ]
    assembled = build_filing_excerpt(chunks, max_input_tokens=80)
    assert len(assembled.segments_kept) < len(chunks)


def test_resolve_model_limits_gpt4o_mini() -> None:
    limits = resolve_model_limits("gpt-4o-mini")
    assert limits.context_window == 128000
