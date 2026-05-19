"""
Enterprise eval regression vs checked-in baseline (S4-05).

Uses offline pipeline only — no paid API keys.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from src.demo.pipeline import run_enterprise_demo
from src.dspy_modules.eval_metrics import evaluate_demo_result
from src.eval.enterprise_eval_schema import SCHEMA_VERSION

_REPO_ROOT = Path(__file__).resolve().parents[2]
BASELINE_PATH = _REPO_ROOT / "docs" / "eval" / "baseline_scores.json"


def _load_baseline() -> dict:
    return json.loads(BASELINE_PATH.read_text(encoding="utf-8"))


@pytest.fixture
def offline_eval_payload():
    demo = run_enterprise_demo(filing_id="acme_corp_10k", offline=True, output_dir=None)
    return evaluate_demo_result(demo)


class TestEnterpriseEvalRegression:
    def test_baseline_file_schema(self):
        baseline = _load_baseline()
        assert baseline["schema_version"] == SCHEMA_VERSION
        assert "aggregate_scores" in baseline
        assert "thresholds" in baseline

    def test_offline_scores_meet_thresholds(self, offline_eval_payload):
        thresholds = offline_eval_payload["thresholds"]
        for name, score in offline_eval_payload["aggregate_scores"].items():
            assert score >= thresholds[name], f"{name}: {score} < {thresholds[name]}"

    def test_no_regression_vs_checked_in_baseline(self, offline_eval_payload):
        baseline = _load_baseline()
        for name, floor in baseline["aggregate_scores"].items():
            actual = offline_eval_payload["aggregate_scores"].get(name, 0.0)
            assert actual >= floor, f"Regression on {name}: {actual} < baseline {floor}"

    def test_all_scenarios_scored(self, offline_eval_payload):
        assert len(offline_eval_payload["per_scenario"]) == 5
        assert offline_eval_payload["all_pass"] is True
