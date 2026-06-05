"""Financial risk cards and eval rubric (S6-08)."""

from __future__ import annotations

import json
from pathlib import Path

from src.scenarios.financial.financial_risk_cards import (
    cards_from_fixture_row,
    eval_rubric_scores,
)

_REPO = Path(__file__).resolve().parents[2]


class TestFinancialRiskEval:
    def test_fixtures_build_cards(self):
        path = _REPO / "data" / "eval" / "financial_risk_fixtures.jsonl"
        rows = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
        all_cards = []
        for row in rows:
            all_cards.extend(cards_from_fixture_row(row))
        assert len(all_cards) >= 2

    def test_rubric_scores_lens_and_regime(self):
        path = _REPO / "data" / "eval" / "financial_risk_fixtures.jsonl"
        row = json.loads(path.read_text().splitlines()[0])
        cards = cards_from_fixture_row(row)
        scores = eval_rubric_scores(cards)
        assert scores["financial_lens_coverage"] >= 0.0
        assert scores["no_advice_claim_penalty"] == 0.0

    def test_measurement_schema_extension(self):
        from src.eval.scenario_measurement_schema import build_measurement_report

        report = build_measurement_report(
            per_theta_slice=[
                {
                    "scenario_type": "financial",
                    "path_mode": "bounded",
                    "theta": {"risk_lens": "credit"},
                    "metrics": {"financial_lens_coverage": 1.0},
                    "n_evaluated": 1,
                }
            ],
            smoke_mode=True,
        )
        assert "financial" in report.per_scenario_type or report.per_theta_slice
