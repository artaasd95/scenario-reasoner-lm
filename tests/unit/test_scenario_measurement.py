"""
Unit tests for scenario measurement harness (S5-04).
"""

from __future__ import annotations

import json
from pathlib import Path

from src.eval.scenario_measurement import run_smoke_measurement
from src.eval.scenario_measurement_schema import (
    SCHEMA_VERSION,
    ScenarioMeasurementReport,
    build_measurement_report,
)


class TestScenarioMeasurementSchema:
    def test_build_report_shape(self):
        report = build_measurement_report(
            per_theta_slice=[
                {
                    "scenario_type": "enterprise",
                    "path_mode": "bounded",
                    "theta": {"num_scenarios": 5},
                    "metrics": {"on_target_composite": 0.8},
                    "n_evaluated": 5,
                },
                {
                    "scenario_type": "causal",
                    "path_mode": "wide",
                    "theta": {"chain_length": 3},
                    "metrics": {"on_target_composite": 0.7},
                    "n_evaluated": 4,
                },
            ],
            goal_preservation={"aggregate": {"goal_preservation": 0.65}},
            smoke_mode=True,
        )
        d = report.to_dict()
        assert d["schema_version"] == SCHEMA_VERSION
        assert "enterprise" in d["per_scenario_type"]
        assert "causal" in d["per_scenario_type"]
        assert len(d["per_theta_slice"]) == 2

    def test_write_json(self, tmp_path):
        report = ScenarioMeasurementReport()
        out = tmp_path / "measurement_report.json"
        report.write_json(out)
        loaded = json.loads(out.read_text())
        assert loaded["schema_version"] == SCHEMA_VERSION


class TestSmokeMeasurement:
    def test_run_smoke_measurement_no_network(self, tmp_path):
        report = run_smoke_measurement(output_dir=tmp_path)
        assert report.metadata.smoke_mode is True
        assert report.metadata.provider_mode == "mock"
        assert len(report.per_theta_slice) >= 4
        out_file = tmp_path / "measurement_report.json"
        assert out_file.exists()

    def test_aggregate_includes_goal_metrics(self):
        report = run_smoke_measurement()
        assert "goal_preservation" in report.aggregate or report.aggregate
