"""
Unit tests for DSPy enterprise risk pipeline modules.
"""

from __future__ import annotations

import json
import pytest
from pathlib import Path

from src.dspy_modules.extract_risks import ExtractRisksModule
from src.dspy_modules.generate_scenarios import GenerateScenariosModule
from src.dspy_modules.verify_scenarios import VerifyScenariosModule
from src.risk.schema import EvidenceChunk, EnterpriseRiskScenarioCard
from src.risk.enterprise_theta import EnterpriseRiskTheta


class TestExtractRisksModule:
    """Tests for evidence extraction module."""

    @pytest.fixture
    def sample_chunks(self):
        """Sample evidence chunks from bundled 10-K."""
        return [
            EvidenceChunk(
                section_name="Risk Factors",
                chunk_id="acme_10k:Risk Factors:0",
                source_span="chars:100-300",
                quote_text="Our top five customers accounted for 48% of net revenue.",
            ),
            EvidenceChunk(
                section_name="Risk Factors",
                chunk_id="acme_10k:Risk Factors:1",
                source_span="chars:350-500",
                quote_text="Custom ASICs sourced from a single foundry partner in Taiwan.",
            ),
            EvidenceChunk(
                section_name="Cybersecurity",
                chunk_id="acme_10k:Cybersecurity:0",
                source_span="chars:1000-1200",
                quote_text="We detected unauthorized access to a staging environment.",
            ),
        ]

    def test_extract_offline_mode(self, sample_chunks):
        """Extract in offline mode (no DSPy)."""
        extractor = ExtractRisksModule(use_dspy=False)
        result = extractor.forward(sample_chunks)
        
        assert "evidence" in result
        assert "chunks" in result
        assert len(result["evidence"]) > 0
        assert len(result["chunks"]) == 3

    def test_extract_output_structure(self, sample_chunks):
        """Verify extraction output structure."""
        extractor = ExtractRisksModule(use_dspy=False)
        result = extractor.forward(sample_chunks)
        
        evidence = result["evidence"]
        assert all("chunk_id" in e for e in evidence)
        assert all("section_name" in e for e in evidence)
        assert all("risk_theme" in e for e in evidence)

    def test_extract_with_trace_callback(self, sample_chunks):
        """Extract with trace callback."""
        traces = []
        
        def callback(stage, inputs, outputs):
            traces.append({"stage": stage, "inputs": inputs, "outputs": outputs})
        
        extractor = ExtractRisksModule(use_dspy=False)
        result = extractor.forward(sample_chunks, trace_callback=callback)
        
        assert len(traces) > 0
        assert traces[0]["stage"] == "extraction"
        assert traces[0]["inputs"]["num_chunks"] == 3


class TestGenerateScenariosModule:
    """Tests for hypothesis generation and scenario building."""

    @pytest.fixture
    def sample_chunks(self):
        """Sample evidence chunks."""
        return [
            EvidenceChunk(
                section_name="Risk Factors",
                chunk_id="acme_10k:Risk Factors:0",
                source_span="chars:100-300",
                quote_text="Supply concentration risk",
            ),
            EvidenceChunk(
                section_name="Cybersecurity",
                chunk_id="acme_10k:Cybersecurity:0",
                source_span="chars:500-700",
                quote_text="Staging environment breach",
            ),
        ]

    @pytest.fixture
    def sample_evidence_payload(self):
        return {
            "evidence": [
                {
                    "chunk_id": "acme_10k:Risk Factors:0",
                    "section_name": "Risk Factors",
                    "quote": "Supply concentration risk",
                    "risk_theme": "concentration",
                }
            ]
        }

    @pytest.fixture
    def sample_theta(self):
        return EnterpriseRiskTheta(
            filing_id="acme_corp_10k",
            num_scenarios=3,
        )

    def test_generate_hypotheses_offline(self, sample_evidence_payload):
        """Generate hypotheses in offline mode."""
        generator = GenerateScenariosModule(use_dspy=False)
        hypotheses = generator.generate_hypotheses(
            sample_evidence_payload,
            company_name="ACME Corporation",
        )
        
        assert len(hypotheses) > 0
        assert all("hypothesis" in h for h in hypotheses)

    def test_generate_hypotheses_structure(self, sample_evidence_payload):
        """Verify hypothesis structure."""
        generator = GenerateScenariosModule(use_dspy=False)
        hypotheses = generator.generate_hypotheses(sample_evidence_payload)
        
        for hyp in hypotheses:
            assert "hypothesis" in hyp
            assert "supporting_chunk_ids" in hyp
            assert "rationale" in hyp

    def test_build_scenarios_offline(self, sample_chunks, sample_theta):
        """Build scenarios in offline mode."""
        builder = GenerateScenariosModule(use_dspy=False)
        # First generate hypotheses
        evidence_payload = {
            "evidence": [
                {
                    "chunk_id": c.chunk_id,
                    "section_name": c.section_name,
                    "quote": c.quote_text[:200],
                    "risk_theme": "general",
                }
                for c in sample_chunks
            ]
        }
        hypotheses = builder.generate_hypotheses(evidence_payload)
        
        # Then build scenarios from hypotheses
        scenarios = builder.build_scenarios(
            hypotheses,
            sample_chunks,
            sample_theta,
        )
        
        assert len(scenarios) > 0
        assert all(isinstance(s, EnterpriseRiskScenarioCard) for s in scenarios)
        assert len(scenarios) <= sample_theta.num_scenarios

    def test_build_scenarios_has_trace_ids(self, sample_chunks, sample_theta):
        """Built scenarios have trace IDs."""
        builder = GenerateScenariosModule(use_dspy=False)
        evidence_payload = {
            "evidence": [
                {
                    "chunk_id": c.chunk_id,
                    "section_name": c.section_name,
                    "quote": c.quote_text,
                    "risk_theme": "general",
                }
                for c in sample_chunks
            ]
        }
        hypotheses = builder.generate_hypotheses(evidence_payload)
        scenarios = builder.build_scenarios(hypotheses, sample_chunks, sample_theta)
        
        assert all(s.trace_id is not None for s in scenarios)

    def test_scenarios_respect_num_scenarios_limit(self, sample_chunks):
        """Built scenarios respect theta.num_scenarios limit."""
        for num in [1, 3, 5]:
            theta = EnterpriseRiskTheta(num_scenarios=num)
            builder = GenerateScenariosModule(use_dspy=False)
            evidence_payload = {
                "evidence": [
                    {
                        "chunk_id": c.chunk_id,
                        "section_name": c.section_name,
                        "quote": c.quote_text,
                        "risk_theme": "general",
                    }
                    for c in sample_chunks
                ]
            }
            hypotheses = builder.generate_hypotheses(evidence_payload)
            scenarios = builder.build_scenarios(hypotheses, sample_chunks, theta)
            
            assert len(scenarios) <= num


class TestVerifyScenariosModule:
    """Tests for scenario critique and ranking."""

    @pytest.fixture
    def sample_cards(self):
        """Sample scenario cards."""
        return [
            EnterpriseRiskScenarioCard(
                title="Scenario 1",
                source_evidence=[
                    EvidenceChunk(
                        section_name="Risk Factors",
                        chunk_id="acme_10k:Risk Factors:0",
                        source_span="chars:100-300",
                        quote_text="Risk text 1",
                    )
                ],
                causal_chain=["Event A", "Event B", "Event C"],
                missed_risk_rationale="Not disclosed",
                severity="catastrophic",
                likelihood="high",
                horizon="6-18 months",
                confidence=0.8,
            ),
            EnterpriseRiskScenarioCard(
                title="Scenario 2",
                source_evidence=[
                    EvidenceChunk(
                        section_name="Cybersecurity",
                        chunk_id="acme_10k:Cybersecurity:0",
                        source_span="chars:500-700",
                        quote_text="Risk text 2",
                    )
                ],
                causal_chain=["Event X", "Event Y"],
                missed_risk_rationale="Understated",
                severity="high",
                likelihood="medium",
                horizon="0-6 months",
                confidence=0.75,
            ),
        ]

    @pytest.fixture
    def sample_evidence_payload(self):
        return {
            "evidence": [
                {
                    "chunk_id": "acme_10k:Risk Factors:0",
                    "section_name": "Risk Factors",
                    "quote": "Risk quote 1",
                    "risk_theme": "supply_chain",
                },
                {
                    "chunk_id": "acme_10k:Cybersecurity:0",
                    "section_name": "Cybersecurity",
                    "quote": "Risk quote 2",
                    "risk_theme": "cyber",
                },
            ]
        }

    def test_critique_offline(self, sample_cards, sample_evidence_payload):
        """Critique scenarios in offline mode."""
        verifier = VerifyScenariosModule(use_dspy=False)
        critiques = verifier.critique(sample_cards, sample_evidence_payload)
        
        assert len(critiques) == len(sample_cards)
        assert all("trace_id" in c for c in critiques)

    def test_critique_structure(self, sample_cards, sample_evidence_payload):
        """Verify critique structure."""
        verifier = VerifyScenariosModule(use_dspy=False)
        critiques = verifier.critique(sample_cards, sample_evidence_payload)
        
        expected_keys = [
            "grounding_score",
            "plausibility_score",
            "severity_clarity",
            "non_duplication",
            "trace_completeness",
        ]
        for critique in critiques:
            for key in expected_keys:
                assert key in critique, f"Missing {key} in critique"

    def test_rank_offline(self, sample_cards, sample_evidence_payload):
        """Rank scenarios in offline mode."""
        verifier = VerifyScenariosModule(use_dspy=False)
        critiques = verifier.critique(sample_cards, sample_evidence_payload)
        ranked = verifier.rank(sample_cards, critiques, strategy="severity_then_likelihood")
        
        assert len(ranked) == len(sample_cards)
        # Catastrophic should typically rank higher than high
        assert ranked[0].severity in ["catastrophic", "high"]

    def test_rank_preserves_all_scenarios(self, sample_cards, sample_evidence_payload):
        """Ranking preserves all scenarios."""
        verifier = VerifyScenariosModule(use_dspy=False)
        critiques = verifier.critique(sample_cards, sample_evidence_payload)
        ranked = verifier.rank(sample_cards, critiques)
        
        original_ids = {c.trace_id for c in sample_cards}
        ranked_ids = {c.trace_id for c in ranked}
        assert original_ids == ranked_ids

    def test_rank_with_trace_callback(self, sample_cards, sample_evidence_payload):
        """Rank with trace callback."""
        traces = []
        
        def callback(stage, inputs, outputs):
            traces.append({"stage": stage})
        
        verifier = VerifyScenariosModule(use_dspy=False)
        critiques = verifier.critique(sample_cards, sample_evidence_payload)
        ranked = verifier.rank(
            sample_cards,
            critiques,
            trace_callback=callback,
        )
        
        assert len(ranked) > 0
        assert any(t["stage"] == "ranking" for t in traces)


class TestPipelineIntegration:
    """Integration tests for the complete pipeline."""

    def test_offline_pipeline_complete(self):
        """Run complete offline pipeline."""
        # Sample chunks
        chunks = [
            EvidenceChunk(
                section_name="Risk Factors",
                chunk_id="acme_10k:Risk Factors:0",
                source_span="chars:100-300",
                quote_text="Supply concentration risk",
            ),
            EvidenceChunk(
                section_name="Cybersecurity",
                chunk_id="acme_10k:Cybersecurity:0",
                source_span="chars:500-700",
                quote_text="Staging breach detected",
            ),
        ]
        
        # Extract
        extractor = ExtractRisksModule(use_dspy=False)
        extract_result = extractor.forward(chunks)
        evidence_payload = extract_result
        
        # Generate hypotheses
        generator = GenerateScenariosModule(use_dspy=False)
        hypotheses = generator.generate_hypotheses(evidence_payload, company_name="ACME")
        
        # Build scenarios
        theta = EnterpriseRiskTheta(num_scenarios=3)
        scenarios = generator.build_scenarios(hypotheses, chunks, theta)
        
        # Verify scenarios
        verifier = VerifyScenariosModule(use_dspy=False)
        critiques = verifier.critique(scenarios, evidence_payload)
        ranked = verifier.rank(scenarios, critiques)
        
        assert len(ranked) == len(scenarios)
        assert all(isinstance(s, EnterpriseRiskScenarioCard) for s in ranked)
        assert all(s.trace_id is not None for s in ranked)
