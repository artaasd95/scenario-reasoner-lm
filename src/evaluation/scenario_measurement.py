"""
Scenario measurability contract implementation (S10-01).

Aligns pipeline measure fields with docs/scenario-measurability-contract.md.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional

from src.data.schemas import ScenarioMeasurement, write_scenario_measurements
from src.eval.scenario_measurement import run_smoke_measurement
from src.eval.scenario_measurement_schema import ScenarioMeasurementReport


CONTRACT_REQUIRED_KEYS = (
    "feasibility",
    "tail_tag",
    "theta_stratum",
    "primary_quality_score",
)


def path_row_to_measurement(row: Dict[str, Any]) -> ScenarioMeasurement:
    """Convert a pipeline row to contract-shaped ScenarioMeasurement."""
    m = row.get("measurement", {})
    return ScenarioMeasurement(
        path_id=str(row.get("path_id", "unknown")),
        feasibility=float(m.get("feasibility", row.get("feasibility", 0.0))),
        tail_tag=str(m.get("tail_tag", row.get("tail_tag", "none"))),
        theta_stratum=str(m.get("theta_stratum", row.get("theta_kind", ""))),
        primary_quality_score=float(m.get("primary_quality_score", 0.0)),
        theta=dict(row.get("theta", {})),
        extra={k: row[k] for k in ("scenario_type", "path_mode") if k in row},
    )


def emit_scenario_measurement_json(
    rows: List[Dict[str, Any]],
    output_path: Path,
) -> None:
    measurements = [path_row_to_measurement(r) for r in rows]
    write_scenario_measurements(measurements, output_path)


def validate_measurement_contract(payload: Dict[str, Any]) -> List[str]:
    """Return list of contract violations (empty if valid)."""
    errors: List[str] = []
    paths = payload.get("paths", [])
    if not paths:
        errors.append("paths must be non-empty")
    for i, p in enumerate(paths):
        for key in CONTRACT_REQUIRED_KEYS:
            if key not in p:
                errors.append(f"paths[{i}] missing {key}")
    return errors


def run_eval_measurement_bundle(
    *,
    output_dir: Path,
    fixtures_path: Optional[Path] = None,
    model_id: str = "",
) -> Dict[str, Any]:
    """
    Run smoke measurement and write scenario_measurement.json + robustness stub.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    report: ScenarioMeasurementReport = run_smoke_measurement(
        fixtures_path=fixtures_path,
        output_dir=None,
        model_id=model_id,
    )
    paths: List[Dict[str, Any]] = []
    for row in report.per_theta_slice:
        paths.append(
            ScenarioMeasurement(
                path_id=str(row.get("fixture_id", row.get("scenario_type", "path"))),
                feasibility=float(row.get("metrics", {}).get("on_target_composite", 0.5)),
                tail_tag="none",
                theta_stratum=str(row.get("scenario_type", "")),
                primary_quality_score=float(row.get("metrics", {}).get("on_target_composite", 0.5)),
                theta=dict(row.get("theta", {})),
            ).to_dict()
        )

    meas_path = output_dir / "scenario_measurement.json"
    payload = {"model_id": model_id or report.metadata.model_id, "paths": paths}
    meas_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    robustness = {
        "aggregate": report.aggregate,
        "per_scenario_type": report.per_scenario_type,
        "schema_version": report.schema_version,
    }
    robustness_path = output_dir / "robustness_report.json"
    robustness_path.write_text(json.dumps(robustness, indent=2), encoding="utf-8")

    errors = validate_measurement_contract(json.loads(meas_path.read_text()))
    return {
        "scenario_measurement": str(meas_path),
        "robustness_report": str(robustness_path),
        "contract_errors": errors,
    }
