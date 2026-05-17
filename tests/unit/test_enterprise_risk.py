"""Tests for enterprise risk schema, ingestion, and offline demo pipeline."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from src.ingestion.chunking import chunk_sections, evidence_chunks_from_sections
from src.ingestion.sec_sections import extract_sections
from src.ingestion.tenk_loader import load_tenk_filing
from src.risk.enterprise_theta import EnterpriseRiskTheta, EnterpriseRiskThetaSampler
from src.risk.schema import EnterpriseRiskScenarioCard, EvidenceChunk
from src.dspy_modules.verify_scenarios import VerifyScenariosModule


@pytest.fixture
def sample_card_dict():
    path = _REPO_ROOT / "tests" / "fixtures" / "sample_scenario_card.json"
    return json.loads(path.read_text(encoding="utf-8"))


def test_evidence_chunk_validation():
    chunk = EvidenceChunk(
        section_name="Risk Factors",
        chunk_id="acme:Risk Factors:0",
        source_span="chars:0-100",
        quote_text="Concentration risk in top customers.",
    )
    assert chunk.chunk_id.startswith("acme")


def test_scenario_card_roundtrip(sample_card_dict):
    card = EnterpriseRiskScenarioCard.from_dict(sample_card_dict)
    restored = EnterpriseRiskScenarioCard.from_dict(card.to_dict())
    assert restored.title == card.title
    assert len(restored.causal_chain) == 3
    assert restored.severity == "catastrophic"


def test_enterprise_theta_demo_default():
    theta = EnterpriseRiskThetaSampler().demo_default()
    assert theta.num_scenarios == 5
    assert theta.filing_id == "acme_corp_10k"


def test_load_bundled_tenk():
    filing = load_tenk_filing("acme_corp_10k")
    assert filing.filing_id == "acme_corp_10k"
    assert "Risk Factors" in filing.raw_text or "risk factors" in filing.raw_text.lower()


def test_extract_sections():
    filing = load_tenk_filing("acme_corp_10k")
    sections = extract_sections(filing.raw_text)
    assert "Risk Factors" in sections
    assert "MD&A" in sections
    assert "Cybersecurity" in sections


def test_chunking_produces_ids():
    filing = load_tenk_filing("acme_corp_10k")
    sections = extract_sections(filing.raw_text)
    chunks = evidence_chunks_from_sections(sections, filing.filing_id)
    assert len(chunks) >= 5
    assert all(c.chunk_id for c in chunks)
    assert all(c.quote_text for c in chunks)


def test_offline_demo_pipeline():
    from src.demo.pipeline import run_enterprise_demo

    result = run_enterprise_demo(filing_id="acme_corp_10k", offline=True, output_dir=None)
    assert len(result["scenarios"]) == 5
    assert result["trace_id"]
    titles = {s["title"] for s in result["scenarios"]}
    assert len(titles) == 5


def test_eval_fixture_load():
    eval_set = VerifyScenariosModule.load_eval_set()
    assert len(eval_set) >= 1
    assert "criteria_thresholds" in eval_set[0]
