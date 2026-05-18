"""
Unit tests for enterprise risk schema and theta.
"""

from __future__ import annotations

import json
import pytest
import uuid
from pathlib import Path

from src.risk.schema import (
    EvidenceChunk,
    EnterpriseRiskScenarioCard,
    ScenarioSeverity,
    scenario_card_from_dict,
)
from src.risk.enterprise_theta import (
    EnterpriseRiskTheta,
    EnterpriseRiskThetaSampler,
)


class TestEvidenceChunk:
    """Tests for EvidenceChunk schema."""

    def test_evidence_chunk_creation(self):
        """Create a valid evidence chunk."""
        chunk = EvidenceChunk(
            section_name="Risk Factors",
            chunk_id="acme_10k:Risk Factors:0",
            source_span="chars:100-400",
            quote_text="Supply chain dependencies create concentration risk.",
        )
        assert chunk.section_name == "Risk Factors"
        assert chunk.chunk_id == "acme_10k:Risk Factors:0"

    def test_evidence_chunk_validates_required_fields(self):
        """Raise ValueError for missing required fields."""
        with pytest.raises(ValueError, match="section_name"):
            EvidenceChunk(
                section_name="",
                chunk_id="id",
                source_span="span",
                quote_text="text",
            )
        
        with pytest.raises(ValueError, match="chunk_id"):
            EvidenceChunk(
                section_name="Risk Factors",
                chunk_id="",
                source_span="span",
                quote_text="text",
            )
        
        with pytest.raises(ValueError, match="quote_text"):
            EvidenceChunk(
                section_name="Risk Factors",
                chunk_id="id",
                source_span="span",
                quote_text="",
            )

    def test_evidence_chunk_to_dict(self):
        """Serialize chunk to dict."""
        chunk = EvidenceChunk(
            section_name="Risk Factors",
            chunk_id="acme_10k:Risk Factors:0",
            source_span="chars:100-400",
            quote_text="Risk text here",
        )
        d = chunk.to_dict()
        assert d["section_name"] == "Risk Factors"
        assert d["quote_text"] == "Risk text here"

    def test_evidence_chunk_from_dict(self):
        """Deserialize chunk from dict."""
        d = {
            "section_name": "Cybersecurity",
            "chunk_id": "acme_10k:Cybersecurity:0",
            "source_span": "chars:500-800",
            "quote_text": "Staging environment breach detected.",
        }
        chunk = EvidenceChunk.from_dict(d)
        assert chunk.section_name == "Cybersecurity"
        assert chunk.quote_text == "Staging environment breach detected."

    def test_evidence_chunk_roundtrip(self):
        """Chunk serializes and deserializes identically."""
        original = EvidenceChunk(
            section_name="Supply Chain",
            chunk_id="acme_10k:Supply Chain:3",
            source_span="chars:1200-1500",
            quote_text="Japanese vendors supply 80% of passive components.",
        )
        restored = EvidenceChunk.from_dict(original.to_dict())
        assert restored.section_name == original.section_name
        assert restored.chunk_id == original.chunk_id
        assert restored.quote_text == original.quote_text


class TestEnterpriseRiskScenarioCard:
    """Tests for EnterpriseRiskScenarioCard schema."""

    @pytest.fixture
    def sample_evidence(self):
        """Sample evidence chunks."""
        return [
            EvidenceChunk(
                section_name="Risk Factors",
                chunk_id="acme_10k:Risk Factors:0",
                source_span="chars:100-300",
                quote_text="Custom ASICs sourced from a single foundry partner in Taiwan.",
            ),
            EvidenceChunk(
                section_name="Risk Factors",
                chunk_id="acme_10k:Risk Factors:1",
                source_span="chars:350-500",
                quote_text="Geopolitical tension could interrupt supply for six months.",
            ),
        ]

    def test_scenario_card_creation(self, sample_evidence):
        """Create a valid scenario card."""
        card = EnterpriseRiskScenarioCard(
            title="Taiwan ASIC Cutoff",
            source_evidence=sample_evidence,
            causal_chain=[
                "Geopolitical tension increases",
                "Taiwan export controls tighten",
                "ASIC supply halts for 6+ months",
                "Production stops",
            ],
            missed_risk_rationale="10-K discloses sole-source but understates cascade.",
            severity="catastrophic",
            likelihood="medium",
            horizon="6-18 months",
            confidence=0.78,
            warning_signals=["Geopolitical news", "Foundry allocation notices"],
            mitigations=["Dual-source qualification"],
        )
        assert card.title == "Taiwan ASIC Cutoff"
        assert len(card.source_evidence) == 2
        assert card.severity == "catastrophic"
        assert card.trace_id is not None

    def test_scenario_card_default_trace_id(self, sample_evidence):
        """Trace ID is auto-generated if not provided."""
        card = EnterpriseRiskScenarioCard(
            title="Test Scenario",
            source_evidence=sample_evidence,
            causal_chain=["Step 1", "Step 2"],
            missed_risk_rationale="Reason here",
            severity="high",
            likelihood="high",
            horizon="6-18 months",
            confidence=0.8,
        )
        assert card.trace_id is not None
        assert len(card.trace_id) > 0

    def test_scenario_card_validates_title(self, sample_evidence):
        """Raise error for empty title."""
        with pytest.raises(ValueError, match="title"):
            EnterpriseRiskScenarioCard(
                title="",
                source_evidence=sample_evidence,
                causal_chain=["Step 1", "Step 2"],
                missed_risk_rationale="Reason",
                severity="high",
                likelihood="high",
                horizon="6-18 months",
                confidence=0.8,
            )

    def test_scenario_card_validates_evidence(self, sample_evidence):
        """Raise error if no evidence provided."""
        with pytest.raises(ValueError, match="source_evidence"):
            EnterpriseRiskScenarioCard(
                title="Test",
                source_evidence=[],
                causal_chain=["Step 1", "Step 2"],
                missed_risk_rationale="Reason",
                severity="high",
                likelihood="high",
                horizon="6-18 months",
                confidence=0.8,
            )

    def test_scenario_card_validates_causal_chain(self, sample_evidence):
        """Raise error if causal chain < 2 steps."""
        with pytest.raises(ValueError, match="causal_chain"):
            EnterpriseRiskScenarioCard(
                title="Test",
                source_evidence=sample_evidence,
                causal_chain=["Only one step"],
                missed_risk_rationale="Reason",
                severity="high",
                likelihood="high",
                horizon="6-18 months",
                confidence=0.8,
            )

    def test_scenario_card_validates_severity(self, sample_evidence):
        """Raise error for invalid severity."""
        with pytest.raises(ValueError, match="severity"):
            EnterpriseRiskScenarioCard(
                title="Test",
                source_evidence=sample_evidence,
                causal_chain=["Step 1", "Step 2"],
                missed_risk_rationale="Reason",
                severity="invalid_level",
                likelihood="high",
                horizon="6-18 months",
                confidence=0.8,
            )

    def test_scenario_card_validates_likelihood(self, sample_evidence):
        """Raise error for invalid likelihood."""
        with pytest.raises(ValueError, match="likelihood"):
            EnterpriseRiskScenarioCard(
                title="Test",
                source_evidence=sample_evidence,
                causal_chain=["Step 1", "Step 2"],
                missed_risk_rationale="Reason",
                severity="high",
                likelihood="impossible",
                horizon="6-18 months",
                confidence=0.8,
            )

    def test_scenario_card_validates_horizon(self, sample_evidence):
        """Raise error for invalid horizon."""
        with pytest.raises(ValueError, match="horizon"):
            EnterpriseRiskScenarioCard(
                title="Test",
                source_evidence=sample_evidence,
                causal_chain=["Step 1", "Step 2"],
                missed_risk_rationale="Reason",
                severity="high",
                likelihood="high",
                horizon="5-10 years",
                confidence=0.8,
            )

    def test_scenario_card_validates_confidence(self, sample_evidence):
        """Raise error for confidence outside [0, 1]."""
        with pytest.raises(ValueError, match="confidence"):
            EnterpriseRiskScenarioCard(
                title="Test",
                source_evidence=sample_evidence,
                causal_chain=["Step 1", "Step 2"],
                missed_risk_rationale="Reason",
                severity="high",
                likelihood="high",
                horizon="6-18 months",
                confidence=1.5,
            )

    def test_scenario_card_to_dict(self, sample_evidence):
        """Serialize card to dict."""
        card = EnterpriseRiskScenarioCard(
            title="Taiwan ASIC Cutoff",
            source_evidence=sample_evidence,
            causal_chain=["Step 1", "Step 2", "Step 3"],
            missed_risk_rationale="Reason",
            severity="catastrophic",
            likelihood="medium",
            horizon="6-18 months",
            confidence=0.78,
        )
        d = card.to_dict()
        assert d["title"] == "Taiwan ASIC Cutoff"
        assert len(d["source_evidence"]) == 2
        assert d["severity"] == "catastrophic"
        assert "trace_id" in d

    def test_scenario_card_to_json(self, sample_evidence):
        """Serialize card to JSON string."""
        card = EnterpriseRiskScenarioCard(
            title="Taiwan ASIC Cutoff",
            source_evidence=sample_evidence,
            causal_chain=["Step 1", "Step 2", "Step 3"],
            missed_risk_rationale="Reason",
            severity="catastrophic",
            likelihood="medium",
            horizon="6-18 months",
            confidence=0.78,
        )
        json_str = card.to_json()
        parsed = json.loads(json_str)
        assert parsed["title"] == "Taiwan ASIC Cutoff"

    def test_scenario_card_from_dict(self, sample_evidence):
        """Deserialize card from dict."""
        d = {
            "title": "Taiwan ASIC Cutoff",
            "source_evidence": [e.to_dict() for e in sample_evidence],
            "causal_chain": ["Step 1", "Step 2", "Step 3"],
            "missed_risk_rationale": "Reason",
            "severity": "catastrophic",
            "likelihood": "medium",
            "horizon": "6-18 months",
            "confidence": 0.78,
            "warning_signals": ["Signal 1"],
            "mitigations": ["Mitigation 1"],
            "trace_id": "trace-123",
        }
        card = EnterpriseRiskScenarioCard.from_dict(d)
        assert card.title == "Taiwan ASIC Cutoff"
        assert len(card.source_evidence) == 2
        assert card.trace_id == "trace-123"

    def test_scenario_card_roundtrip(self, sample_evidence):
        """Card serializes and deserializes identically."""
        original = EnterpriseRiskScenarioCard(
            title="Taiwan ASIC Cutoff",
            source_evidence=sample_evidence,
            causal_chain=["Step 1", "Step 2", "Step 3"],
            missed_risk_rationale="Reason",
            severity="catastrophic",
            likelihood="medium",
            horizon="6-18 months",
            confidence=0.78,
            warning_signals=["Signal 1"],
            mitigations=["Mitigation 1"],
        )
        restored = EnterpriseRiskScenarioCard.from_dict(original.to_dict())
        assert restored.title == original.title
        assert restored.severity == original.severity
        assert len(restored.source_evidence) == len(original.source_evidence)


class TestEnterpriseRiskTheta:
    """Tests for EnterpriseRiskTheta parameter space."""

    def test_theta_creation_defaults(self):
        """Create theta with default parameters."""
        theta = EnterpriseRiskTheta()
        assert theta.filing_id == "acme_corp_10k"
        assert theta.num_scenarios == 5
        assert theta.severity_floor == "high"
        assert len(theta.focus_sections) > 0

    def test_theta_creation_custom(self):
        """Create theta with custom parameters."""
        theta = EnterpriseRiskTheta(
            filing_id="custom_filing",
            num_scenarios=3,
            severity_floor="medium",
            critique_passes=2,
        )
        assert theta.filing_id == "custom_filing"
        assert theta.num_scenarios == 3
        assert theta.severity_floor == "medium"
        assert theta.critique_passes == 2

    def test_theta_validates_filing_id(self):
        """Raise error for empty filing ID."""
        with pytest.raises(ValueError, match="filing_id"):
            EnterpriseRiskTheta(filing_id="")

    def test_theta_validates_num_scenarios(self):
        """Raise error for invalid num_scenarios."""
        with pytest.raises(ValueError, match="num_scenarios"):
            EnterpriseRiskTheta(num_scenarios=0)
        
        with pytest.raises(ValueError, match="num_scenarios"):
            EnterpriseRiskTheta(num_scenarios=11)

    def test_theta_validates_severity_floor(self):
        """Raise error for invalid severity floor."""
        with pytest.raises(ValueError, match="severity_floor"):
            EnterpriseRiskTheta(severity_floor="extreme")

    def test_theta_validates_critique_passes(self):
        """Raise error for negative critique passes."""
        with pytest.raises(ValueError, match="critique_passes"):
            EnterpriseRiskTheta(critique_passes=-1)

    def test_theta_validates_ranking_strategy(self):
        """Raise error for invalid ranking strategy."""
        with pytest.raises(ValueError, match="ranking_strategy"):
            EnterpriseRiskTheta(ranking_strategy="unknown_strategy")

    def test_theta_validates_focus_sections(self):
        """Raise error for empty focus sections."""
        with pytest.raises(ValueError, match="focus_sections"):
            EnterpriseRiskTheta(focus_sections=())

    def test_theta_to_dict(self):
        """Serialize theta to dict."""
        theta = EnterpriseRiskTheta(
            filing_id="acme_10k",
            num_scenarios=5,
            severity_floor="high",
        )
        d = theta.to_dict()
        assert d["filing_id"] == "acme_10k"
        assert d["num_scenarios"] == 5
        assert "focus_sections" in d


class TestEnterpriseRiskThetaSampler:
    """Tests for parameter space sampling."""

    def test_sampler_defaults(self):
        """Create sampler with defaults."""
        sampler = EnterpriseRiskThetaSampler()
        assert len(sampler.filing_ids) > 0
        assert len(sampler.severity_floors) > 0

    def test_sampler_single_sample(self):
        """Draw one sample from sampler."""
        sampler = EnterpriseRiskThetaSampler(seed=42)
        theta = sampler.sample()
        assert isinstance(theta, EnterpriseRiskTheta)
        assert 1 <= theta.num_scenarios <= 10

    def test_sampler_deterministic(self):
        """Same seed produces same samples."""
        sampler1 = EnterpriseRiskThetaSampler(seed=42)
        sampler2 = EnterpriseRiskThetaSampler(seed=42)
        
        theta1 = sampler1.sample()
        theta2 = sampler2.sample()
        
        assert theta1.filing_id == theta2.filing_id
        assert theta1.num_scenarios == theta2.num_scenarios

    def test_sampler_custom_ranges(self):
        """Sample with custom ranges."""
        sampler = EnterpriseRiskThetaSampler(
            filing_ids=["test_filing"],
            num_scenarios_range=(2, 4),
            severity_floors=["medium", "high"],
            seed=42,
        )
        for _ in range(10):
            theta = sampler.sample()
            assert 2 <= theta.num_scenarios <= 4
            assert theta.severity_floor in ["medium", "high"]
            assert theta.filing_id == "test_filing"
