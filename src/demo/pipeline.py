"""
Enterprise risk 10-K demo pipeline.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Optional

from src.dspy_modules.extract_risks import ExtractRisksModule
from src.dspy_modules.generate_scenarios import GenerateScenariosModule
from src.dspy_modules.verify_scenarios import VerifyScenariosModule
from src.ingestion.chunking import evidence_chunks_from_sections
from src.ingestion.sec_sections import extract_sections
from src.ingestion.tenk_loader import load_tenk_filing
from src.dspy_modules.optimize import OPTIMIZER_BOOTSTRAP, OptimizerConfig, resolve_optimizer
from src.risk.enterprise_theta import EnterpriseRiskTheta
from src.tracing.trace_context import TenKDemoTrace, TraceSpanName


def run_enterprise_demo(
    filing_id: str = "acme_corp_10k",
    offline: bool = True,
    output_dir: Optional[str] = None,
    optimizer_config: Optional[OptimizerConfig] = None,
) -> dict:
    theta = EnterpriseRiskTheta(filing_id=filing_id)
    opt = optimizer_config or resolve_optimizer(OPTIMIZER_BOOTSTRAP)
    use_dspy = not offline and os.getenv("LLM_PROVIDER", "offline") != "offline"
    trace = TenKDemoTrace(filing_id=filing_id)

    def _cb(stage, inputs, outputs):
        trace.trace_callback(stage, inputs, outputs)

    filing = trace.run_stage(
        TraceSpanName.LOADING,
        lambda: load_tenk_filing(filing_id),
        inputs={"filing_id": filing_id},
    )

    sections = trace.run_stage(
        TraceSpanName.CHUNKING,
        lambda: extract_sections(
            filing.raw_text,
            section_names=list(theta.focus_sections),
        ),
        inputs={"sections_requested": list(theta.focus_sections)},
    )

    chunks = evidence_chunks_from_sections(
        sections,
        filing_id=filing.filing_id,
        focus_sections=list(theta.focus_sections),
    )
    trace.trace_callback(
        "chunking_complete",
        {"num_sections": len(sections)},
        {"num_chunks": len(chunks), "evidence_chunk_ids": [c.chunk_id for c in chunks]},
    )

    extractor = ExtractRisksModule(use_dspy=use_dspy)
    evidence_payload = trace.run_stage(
        TraceSpanName.EXTRACTION,
        lambda: extractor.forward(chunks, trace_callback=_cb),
        inputs={"num_chunks": len(chunks)},
    )

    generator = GenerateScenariosModule(use_dspy=use_dspy)
    hypotheses = trace.run_stage(
        TraceSpanName.HYPOTHESES,
        lambda: generator.generate_hypotheses(
            evidence_payload,
            company_name=filing.company_name,
            trace_callback=_cb,
        ),
        inputs={"company": filing.company_name},
    )

    cards = generator.build_scenarios(
        hypotheses,
        chunks,
        theta,
        trace_callback=_cb,
    )

    verifier = VerifyScenariosModule(use_dspy=use_dspy)
    critiques = trace.run_stage(
        TraceSpanName.CRITIQUE,
        lambda: verifier.critique(cards, evidence_payload, trace_callback=_cb),
        inputs={"num_cards": len(cards)},
    )

    ranked = trace.run_stage(
        TraceSpanName.RANKING,
        lambda: verifier.rank(
            cards,
            critiques,
            strategy=theta.ranking_strategy,
            trace_callback=_cb,
        ),
        inputs={"strategy": theta.ranking_strategy},
    )

    result = {
        "filing_id": filing.filing_id,
        "trace_id": trace.trace_id,
        "langfuse_url": trace.langfuse_url,
        "theta": theta.to_dict(),
        "optimizer": opt.name,
        "optimizer_seed": opt.seed,
        "token_budget_note": opt.token_budget_note,
        "num_chunks": len(chunks),
        "scenarios": [c.to_dict() for c in ranked],
        "critiques": critiques,
    }

    if output_dir:
        out = Path(output_dir)
        out.mkdir(parents=True, exist_ok=True)
        trace.run_stage(
            TraceSpanName.RENDERING,
            lambda: _write_outputs(out, result),
            inputs={"output_dir": str(out)},
        )

    trace.flush()
    return result


def _write_outputs(out: Path, result: dict) -> dict:
    path = out / "demo_result.json"
    path.write_text(json.dumps(result, indent=2), encoding="utf-8")
    return {"written": str(path)}
