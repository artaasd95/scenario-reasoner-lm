"""
Unit tests for scenario-set reasoning eval harness (S6-05 / S7-05).
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from src.eval.scenario_reasoning_eval import run_full_reasoning_eval, run_smoke_reasoning_eval
from src.eval.scenario_reasoning_eval_schema import (
    REASONING_EVAL_SCHEMA_VERSION,
    ScenarioReasoningEvalReport,
    build_reasoning_eval_report,
)
from src.scenarios.resource_gate import assert_execution_sprint_allowed


class TestScenarioReasoningEvalSchema:
    def test_build_report_shape(self):
        report = build_reasoning_eval_report(
            per_theta_slice=[
                {
                    "path_mode": "enumerated",
                    "scenario_type": "enterprise",
                    "theta": {"num_scenarios": 5},
                    "metrics": {"on_target_composite": 0.8},
                    "path_count": 5,
                },
                {
                    "path_mode": "exploratory",
                    "scenario_type": "causal",
                    "theta": {"chain_length": 3},
                    "metrics": {"on_target_composite": 0.7},
                    "path_count": 8,
                },
            ],
            goal_preservation={"aggregate": {"goal_preservation": 0.65}},
            coherence={"aggregate": {"cross_scenario_coherence": 0.72}},
            smoke_mode=True,
        )
        d = report.to_dict()
        assert d["schema_version"] == REASONING_EVAL_SCHEMA_VERSION
        assert "enumerated" in d["per_path_type"]
        assert "exploratory" in d["per_path_type"]
        assert len(d["per_theta_slice"]) == 2

    def test_write_json(self, tmp_path):
        report = ScenarioReasoningEvalReport()
        out = tmp_path / "reasoning_eval_report.json"
        report.write_json(out)
        loaded = json.loads(out.read_text())
        assert loaded["schema_version"] == REASONING_EVAL_SCHEMA_VERSION


class TestSmokeReasoningEval:
    def test_run_smoke_no_network(self, tmp_path):
        report = run_smoke_reasoning_eval(output_dir=tmp_path)
        assert report.metadata.smoke_mode is True
        assert report.metadata.execution_sprint is False
        assert report.metadata.provider_mode == "mock"
        assert len(report.per_theta_slice) >= 6
        assert "enumerated" in report.per_path_type
        assert "exploratory" in report.per_path_type
        out_file = tmp_path / "reasoning_eval_report.json"
        assert out_file.exists()

    def test_aggregate_includes_goal_and_coherence(self):
        report = run_smoke_reasoning_eval()
        assert "goal_preservation" in report.aggregate or report.aggregate
        assert "cross_scenario_coherence" in report.aggregate or report.aggregate


class TestExecutionSprintGate:
    def test_full_pipeline_blocked_without_gate(self, monkeypatch):
        monkeypatch.delenv("EXECUTION_SPRINT_GATE", raising=False)
        with pytest.raises(RuntimeError, match="EXECUTION_SPRINT_GATE"):
            assert_execution_sprint_allowed(full_pipeline=True)

    def test_full_pipeline_blocked_in_eval(self, monkeypatch):
        monkeypatch.delenv("EXECUTION_SPRINT_GATE", raising=False)
        with pytest.raises(RuntimeError, match="EXECUTION_SPRINT_GATE"):
            run_full_reasoning_eval()
