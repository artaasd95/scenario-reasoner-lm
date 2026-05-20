"""
Multi-scenario measurement harness (S5-04).

Smoke mode: bundled fixtures only, no paid API keys.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional

from src.eval.scenario_measurement_schema import (
    ScenarioMeasurementReport,
    build_measurement_report,
)
from src.metrics.goal_preservation_metrics import evaluate_fixtures
from src.scenarios.simulation_runner import ScenarioSimulationRunner, load_simulation_fixtures
from src.scenarios.theta_mapping import enterprise_theta_to_causal_slice
from src.risk.enterprise_theta import EnterpriseRiskTheta
from src.metrics.goal_preservation_metrics import score_goal_alignment


def _metrics_for_simulation_result(result: Dict[str, Any]) -> Dict[str, float]:
    """Derive lightweight measurement scores from a simulation result."""
    goal = result.get("goal", "")
    spans = result.get("spans", [])
    span_text = " ".join(s.get("name", "") for s in spans)
    theta = result.get("theta") or (result.get("paths") or [{}])[0]
    if isinstance(theta, list):
        theta = theta[0] if theta else {}
    if result.get("scenario_type") == "enterprise" and theta:
        try:
            et = EnterpriseRiskTheta(**{k: v for k, v in theta.items() if k != "focus_sections"})
            theta = et.to_dict()
            theta["causal_slice"] = enterprise_theta_to_causal_slice(et).to_dict()
        except (TypeError, ValueError):
            pass
    scores = score_goal_alignment(span_text or "simulation", goal, theta if isinstance(theta, dict) else {})
    path_count = result.get("path_count") or result.get("scenario_count") or 1
    return {
        **scores,
        "path_coverage": min(path_count / 5.0, 1.0),
    }


def run_smoke_measurement(
    *,
    fixtures_path: Optional[Path] = None,
    output_dir: Optional[Path] = None,
) -> ScenarioMeasurementReport:
    """
    Run measurement on bundled fixtures only (no network, no paid keys).
    """
    runner = ScenarioSimulationRunner(fixtures_path=fixtures_path, dry_run=True)
    sim_report = runner.run_all()
    per_theta_slice: List[Dict[str, Any]] = []

    for result in sim_report["results"]:
        metrics = _metrics_for_simulation_result(result)
        theta = result.get("theta")
        if not theta and result.get("paths"):
            theta = result["paths"][0].get("theta", {})
        per_theta_slice.append(
            {
                "scenario_type": result.get("scenario_type"),
                "path_mode": result.get("path_mode"),
                "theta": theta or {},
                "metrics": metrics,
                "fixture_id": result.get("trace_id", ""),
                "n_evaluated": result.get("path_count") or result.get("scenario_count") or 1,
            }
        )

    goal_eval = evaluate_fixtures()
    report = build_measurement_report(
        per_theta_slice=per_theta_slice,
        goal_preservation=goal_eval,
        smoke_mode=True,
        provider_mode="mock",
    )

    if output_dir:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        report.write_json(output_dir / "measurement_report.json")

    return report
