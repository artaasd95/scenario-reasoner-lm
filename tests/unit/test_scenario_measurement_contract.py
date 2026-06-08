"""Unit tests for scenario measurement contract (SR-08)."""

from __future__ import annotations

import json
from pathlib import Path

from src.evaluation.scenario_measurement import (
    run_eval_measurement_bundle,
    validate_measurement_contract,
)


class TestScenarioMeasurementContract:
    def test_validate_valid_payload(self):
        payload = {
            "paths": [{
                "feasibility": 0.9,
                "tail_tag": "none",
                "theta_stratum": "causal",
                "primary_quality_score": 0.85,
            }],
        }
        assert validate_measurement_contract(payload) == []

    def test_validate_missing_fields(self):
        errors = validate_measurement_contract({"paths": [{}]})
        assert any("feasibility" in e for e in errors)

    def test_run_eval_measurement_bundle_writes_file(self, tmp_path: Path):
        result = run_eval_measurement_bundle(output_dir=tmp_path, model_id="smoke")
        meas_path = Path(result["scenario_measurement"])
        assert meas_path.is_file()
        payload = json.loads(meas_path.read_text(encoding="utf-8"))
        assert validate_measurement_contract(payload) == []
