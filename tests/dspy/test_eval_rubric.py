"""
Unit tests for eval loader and rubric parsing (S4-02).

No live provider calls.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from src.dspy_modules.eval_metrics import (
    EvalRecord,
    aggregate_criterion_scores,
    load_enterprise_eval_set,
    parse_rubric_thresholds,
    score_grounding,
    score_severity_clarity,
)
from src.dspy_modules.signatures import EVAL_CRITERIA
from src.risk.schema import EnterpriseRiskScenarioCard, EvidenceChunk

_REPO_ROOT = Path(__file__).resolve().parents[2]
EVAL_PATH = _REPO_ROOT / "data" / "eval" / "enterprise_risk_tiny.jsonl"


class TestEvalLoader:
    def test_load_jsonl_row_count(self):
        records = load_enterprise_eval_set(EVAL_PATH)
        assert len(records) == 5

    def test_eval_record_fields(self):
        records = load_enterprise_eval_set(EVAL_PATH)
        row = records[0]
        assert row.filing_id == "acme_corp_10k"
        assert row.scenario_index == 0
        assert "Taiwan" in row.expected_title_keywords
        assert row.reviewer_notes
        assert row.failure_mode_tags
        assert row.rubric_hints

    def test_jsonl_lines_parse(self):
        for line in EVAL_PATH.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            data = json.loads(line)
            rec = EvalRecord.from_dict(data)
            assert rec.eval_criteria
            for name in EVAL_CRITERIA:
                assert name in rec.eval_criteria


class TestRubricThresholds:
    def test_parse_rubric_thresholds(self):
        records = load_enterprise_eval_set(EVAL_PATH)
        thresholds = parse_rubric_thresholds(records)
        assert thresholds["grounding"] == 0.7
        assert thresholds["non_duplication"] == 0.9

    def test_aggregate_empty(self):
        agg = aggregate_criterion_scores([])
        assert all(v == 0.0 for v in agg.values())


class TestScoringHelpers:
    @pytest.fixture
    def sample_card(self):
        return EnterpriseRiskScenarioCard(
            title="Single-foundry ASIC cutoff after Taiwan disruption",
            source_evidence=[
                EvidenceChunk(
                    section_name="Risk Factors",
                    chunk_id="rf-1",
                    source_span="1-200",
                    quote_text="Taiwan concentration risk",
                )
            ],
            causal_chain=["a", "b", "c"],
            missed_risk_rationale="Understated foundry risk",
            severity="catastrophic",
            likelihood="medium",
            horizon="6-18 months",
            confidence=0.8,
            warning_signals=[],
            mitigations=[],
            trace_id="test-trace",
        )

    def test_score_grounding_pass(self, sample_card):
        record = EvalRecord(
            filing_id="acme_corp_10k",
            scenario_index=0,
            expected_title_keywords=["Taiwan", "ASIC"],
            expected_severity="catastrophic",
            golden_title_substring="Taiwan",
        )
        assert score_grounding(sample_card, record) >= 0.9

    def test_score_severity_clarity_match(self, sample_card):
        record = EvalRecord(
            filing_id="acme_corp_10k",
            scenario_index=0,
            expected_severity="catastrophic",
        )
        assert score_severity_clarity(sample_card, record) >= 0.9
