"""
Evidence extraction module (DSPy or offline stub).
"""

from __future__ import annotations

import json
from typing import List, Optional

from src.dspy_modules.filing_context import build_filing_excerpt
from src.dspy_modules.signatures import ExtractEvidence, _require_dspy
from src.risk.schema import EvidenceChunk


class ExtractRisksModule:
    """Extract risk themes and structured evidence from chunks."""

    def __init__(self, use_dspy: bool = True) -> None:
        self.use_dspy = use_dspy
        self._predictor = None
        if use_dspy:
            _require_dspy()
            import dspy

            self._predictor = dspy.Predict(ExtractEvidence)

    def forward(
        self,
        chunks: List[EvidenceChunk],
        trace_callback: Optional[callable] = None,
    ) -> dict:
        assembled = build_filing_excerpt(chunks)
        excerpt = assembled.text
        kept_sections = {
            c.section_name
            for c in chunks
            if c.chunk_id in assembled.segments_kept
        }
        sections = ", ".join(sorted(kept_sections))

        if self._predictor is not None:
            import dspy

            result = self._predictor(filing_excerpt=excerpt, section_names=sections)
            payload = json.loads(result.evidence_json)
        else:
            payload = self._offline_extract(chunks)

        if trace_callback:
            trace_callback(
                stage="extraction",
                inputs={"num_chunks": len(chunks), "sections": sections},
                outputs={"num_items": len(payload)},
            )
        return {"evidence": payload, "chunks": [c.to_dict() for c in chunks]}

    @staticmethod
    def _offline_extract(chunks: List[EvidenceChunk]) -> List[dict]:
        themes = {
            "Risk Factors": "concentration_and_supply",
            "MD&A": "margin_and_liquidity",
            "Legal Proceedings": "regulatory_and_litigation",
            "Cybersecurity": "cyber_incident",
            "Regulatory": "compliance_deadline",
            "Supply Chain": "sole_source_lead_time",
            "Business": "customer_concentration",
        }
        out = []
        for c in chunks:
            out.append(
                {
                    "chunk_id": c.chunk_id,
                    "section_name": c.section_name,
                    "quote": c.quote_text[:400],
                    "risk_theme": themes.get(c.section_name, "general"),
                }
            )
        return out
