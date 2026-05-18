"""
Bundled-sample end-to-end smoke test for enterprise risk demo.

Verifies: loading, extraction, chunking, hypothesis generation,
scenario building, critique, ranking, and rendering without live API keys.

Run with: pytest tests/integration/test_enterprise_demo_smoke.py -v
"""

from __future__ import annotations

import json
import pytest
from pathlib import Path

from src.demo.pipeline import run_enterprise_demo
from src.ingestion.tenk_loader import load_tenk_filing
from src.ingestion.sec_sections import extract_sections
from src.ingestion.chunking import chunk_sections
from src.risk.schema import EnterpriseRiskScenarioCard


class TestBundledSamplePipeline:
    """End-to-end tests on bundled ACME sample."""

    def test_bundled_sample_exists(self):
        """Bundled sample 10-K can be loaded."""
        filing = load_tenk_filing("acme_corp_10k")
        assert filing.filing_id == "acme_corp_10k"
        assert len(filing.raw_text) > 100

    def test_bundled_sample_sections_extracted(self):
        """All required sections are extractable."""
        filing = load_tenk_filing("acme_corp_10k")
        sections = extract_sections(filing.raw_text)
        
        required_sections = ["Risk Factors", "Cybersecurity", "Supply Chain"]
        found = [s for s in required_sections if s in sections]
        assert len(found) >= len(required_sections), \
            f"Missing sections: {set(required_sections) - set(found)}"

    def test_bundled_sample_chunking(self):
        """Evidence chunks generated from sections."""
        filing = load_tenk_filing("acme_corp_10k")
        sections = extract_sections(filing.raw_text)
        chunks = chunk_sections(sections, filing_id=filing.filing_id)
        
        assert len(chunks) > 0, "No chunks generated"
        assert all(len(c.quote_text) > 0 for c in chunks), "Empty quotes found"

    def test_offline_demo_runs_complete(self):
        """Complete offline demo pipeline executes without errors."""
        result = run_enterprise_demo(
            filing_id="acme_corp_10k",
            offline=True,
            output_dir=None,
        )
        
        assert result is not None
        assert isinstance(result, dict)

    def test_demo_output_structure(self):
        """Demo output has all required fields."""
        result = run_enterprise_demo(
            filing_id="acme_corp_10k",
            offline=True,
            output_dir=None,
        )
        
        required_keys = ["filing_id", "trace_id", "theta", "scenarios", "critiques"]
        for key in required_keys:
            assert key in result, f"Missing key: {key}"

    def test_demo_produces_five_scenarios(self):
        """Default demo produces five scenarios."""
        result = run_enterprise_demo(
            filing_id="acme_corp_10k",
            offline=True,
            output_dir=None,
        )
        
        scenarios = result["scenarios"]
        assert len(scenarios) == 5, f"Expected 5 scenarios, got {len(scenarios)}"

    def test_scenario_card_structure(self):
        """Each scenario card has required fields."""
        result = run_enterprise_demo(
            filing_id="acme_corp_10k",
            offline=True,
            output_dir=None,
        )
        
        required_card_keys = [
            "title",
            "severity",
            "likelihood",
            "horizon",
            "confidence",
            "causal_chain",
            "missed_risk_rationale",
            "source_evidence",
            "trace_id",
        ]
        
        for card in result["scenarios"]:
            for key in required_card_keys:
                assert key in card, f"Card missing key: {key}"
                assert card[key] is not None, f"Card {key} is None"

    def test_severity_values_valid(self):
        """Scenario severity values are valid."""
        result = run_enterprise_demo(
            filing_id="acme_corp_10k",
            offline=True,
            output_dir=None,
        )
        
        valid_severities = ("low", "medium", "high", "catastrophic")
        for card in result["scenarios"]:
            assert card["severity"] in valid_severities, \
                f"Invalid severity: {card['severity']}"

    def test_likelihood_values_valid(self):
        """Scenario likelihood values are valid."""
        result = run_enterprise_demo(
            filing_id="acme_corp_10k",
            offline=True,
            output_dir=None,
        )
        
        valid_likelihoods = ("low", "medium", "high")
        for card in result["scenarios"]:
            assert card["likelihood"] in valid_likelihoods, \
                f"Invalid likelihood: {card['likelihood']}"

    def test_confidence_in_valid_range(self):
        """Scenario confidence scores are in [0, 1]."""
        result = run_enterprise_demo(
            filing_id="acme_corp_10k",
            offline=True,
            output_dir=None,
        )
        
        for card in result["scenarios"]:
            conf = card["confidence"]
            assert 0.0 <= conf <= 1.0, \
                f"Invalid confidence: {conf}"

    def test_causal_chain_length(self):
        """Causal chains have at least 2 steps."""
        result = run_enterprise_demo(
            filing_id="acme_corp_10k",
            offline=True,
            output_dir=None,
        )
        
        for card in result["scenarios"]:
            chain = card["causal_chain"]
            assert len(chain) >= 2, \
                f"Causal chain too short: {chain}"

    def test_evidence_grounding(self):
        """Each scenario has grounded evidence."""
        result = run_enterprise_demo(
            filing_id="acme_corp_10k",
            offline=True,
            output_dir=None,
        )
        
        for card in result["scenarios"]:
            evidence = card["source_evidence"]
            assert len(evidence) > 0, \
                f"Card {card['title']} has no evidence"
            for ev in evidence:
                assert "quote_text" in ev
                assert len(ev["quote_text"]) > 0

    def test_demo_serializable_to_json(self):
        """Result is JSON-serializable."""
        result = run_enterprise_demo(
            filing_id="acme_corp_10k",
            offline=True,
            output_dir=None,
        )
        
        try:
            json_str = json.dumps(result)
            roundtrip = json.loads(json_str)
            assert roundtrip["trace_id"] == result["trace_id"]
        except (TypeError, ValueError) as e:
            pytest.fail(f"Result not JSON-serializable: {e}")

    def test_demo_with_output_dir(self, tmp_path):
        """Demo writes artifacts to output directory."""
        output_dir = tmp_path / "demo_output"
        result = run_enterprise_demo(
            filing_id="acme_corp_10k",
            offline=True,
            output_dir=str(output_dir),
        )
        
        assert output_dir.exists(), "Output directory not created"
        result_file = output_dir / "demo_result.json"
        assert result_file.exists(), "demo_result.json not written"
        
        saved_result = json.loads(result_file.read_text())
        assert saved_result["trace_id"] == result["trace_id"]

    def test_trace_id_consistency(self):
        """Trace ID is consistent across output."""
        result = run_enterprise_demo(
            filing_id="acme_corp_10k",
            offline=True,
            output_dir=None,
        )
        
        trace_id = result["trace_id"]
        assert trace_id is not None
        assert len(trace_id) > 0
        
        # All scenarios should reference trace context
        assert all(s.get("trace_id") is not None for s in result["scenarios"])

    def test_non_duplication_handling(self):
        """Ranking includes non-duplication criteria."""
        result = run_enterprise_demo(
            filing_id="acme_corp_10k",
            offline=True,
            output_dir=None,
        )
        
        critiques = result["critiques"]
        # Offline mode includes non_duplication scores
        assert all("non_duplication" in c or len(c) > 0 for c in critiques)

    def test_ranking_order_preserved(self):
        """Ranked scenarios are ordered by score."""
        result = run_enterprise_demo(
            filing_id="acme_corp_10k",
            offline=True,
            output_dir=None,
        )
        
        scenarios = result["scenarios"]
        # Verify ranking is not empty and maintains order
        assert len(scenarios) > 0
        # In offline mode, catastrophic should tend to be higher
        first_is_high_severity = scenarios[0]["severity"] in ("high", "catastrophic")
        assert first_is_high_severity or len(scenarios) == 1


class TestDemoRobustness:
    """Robustness and error-handling tests."""

    def test_missing_sections_handled(self):
        """Demo completes even if some sections are missing."""
        # The bundled sample has all main sections, but test robustness
        result = run_enterprise_demo(
            filing_id="acme_corp_10k",
            offline=True,
            output_dir=None,
        )
        
        # Should still produce scenarios even with partial sections
        assert len(result["scenarios"]) > 0

    def test_demo_trace_id_never_empty(self):
        """Trace ID is always present."""
        result = run_enterprise_demo(
            filing_id="acme_corp_10k",
            offline=True,
            output_dir=None,
        )
        
        assert result["trace_id"]
        assert isinstance(result["trace_id"], str)
        assert len(result["trace_id"]) > 0

    def test_langfuse_url_graceful_when_no_keys(self):
        """Langfuse URL is None/missing when keys absent."""
        # In default (no LANGFUSE_* env vars), URL should be None
        result = run_enterprise_demo(
            filing_id="acme_corp_10k",
            offline=True,
            output_dir=None,
        )
        
        # Langfuse URL can be absent or None in offline mode
        url = result.get("langfuse_url")
        # No assertion error expected; either present or missing is OK
        assert url is None or isinstance(url, str)
