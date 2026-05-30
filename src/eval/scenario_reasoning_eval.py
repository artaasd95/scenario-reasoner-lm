"""
Scenario-set reasoning eval harness (S6-05 / S7-05).

Smoke mode: bundled fixtures only, no paid API keys.
Full pipeline execution requires execution-sprint resource gate.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional

from src.eval.scenario_reasoning_eval_schema import (
    ScenarioReasoningEvalReport,
    build_reasoning_eval_report,
)
from src.metrics.cross_scenario_coherence_metrics import evaluate_fixtures as evaluate_coherence
from src.metrics.goal_preservation_metrics import evaluate_fixtures as evaluate_goal_preservation
from src.scenarios.enumerated_path_generator import generate_from_fixture, load_enumerated_fixtures
from src.scenarios.exploratory_path_generator import (
    generate_from_exploratory_fixture,
    load_exploratory_fixtures,
)
from src.scenarios.resource_gate import assert_execution_sprint_allowed, effective_provider_mode
from src.scenarios.scenario_reasoning_batch import ScenarioReasoningBatch


def _metrics_for_batch(batch: Dict[str, Any]) -> Dict[str, float]:
    results = batch.get("results", [])
    if not results:
        return {"on_target_composite": 0.0, "path_coverage": 0.0}
    composites = [r["goal_scores"]["on_target_composite"] for r in results]
    return {
        "on_target_composite": round(sum(composites) / len(composites), 4),
        "path_coverage": min(batch.get("path_count", 0) / 5.0, 1.0),
        "on_target_rate": round(batch.get("on_target_count", 0) / max(len(results), 1), 4),
    }


def run_smoke_reasoning_eval(
    *,
    output_dir: Optional[Path] = None,
    seed: int = 42,
) -> ScenarioReasoningEvalReport:
    """Run reasoning eval on bundled fixtures only (no network, no paid keys)."""
    batch_runner = ScenarioReasoningBatch(dry_run=True, seed=seed)
    per_theta_slice: List[Dict[str, Any]] = []

    for fx in load_enumerated_fixtures():
        paths = generate_from_fixture(fx, seed=seed)
        batch = batch_runner.run_paths(paths)
        batch["path_mode"] = "enumerated"
        batch["fixture_id"] = fx.fixture_id
        batch["scenario_type"] = fx.scenario_type.value
        theta = fx.theta
        per_theta_slice.append(
            {
                "path_mode": "enumerated",
                "scenario_type": fx.scenario_type.value,
                "theta": theta,
                "metrics": _metrics_for_batch(batch),
                "path_count": batch["path_count"],
                "fixture_id": fx.fixture_id,
            }
        )

    for fx in load_exploratory_fixtures():
        paths = generate_from_exploratory_fixture(fx)
        batch = batch_runner.run_paths(paths)
        batch["path_mode"] = "exploratory"
        batch["fixture_id"] = fx.fixture_id
        batch["scenario_type"] = fx.scenario_type
        per_theta_slice.append(
            {
                "path_mode": "exploratory",
                "scenario_type": fx.scenario_type,
                "theta": fx.theta_anchor,
                "metrics": _metrics_for_batch(batch),
                "path_count": batch["path_count"],
                "fixture_id": fx.fixture_id,
            }
        )

    goal_eval = evaluate_goal_preservation()
    coherence_eval = evaluate_coherence()
    report = build_reasoning_eval_report(
        per_theta_slice=per_theta_slice,
        goal_preservation=goal_eval,
        coherence=coherence_eval,
        smoke_mode=True,
        execution_sprint=False,
        provider_mode="mock",
    )

    if output_dir:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        report.write_json(output_dir / "reasoning_eval_report.json")

    return report


def run_full_reasoning_eval(
    *,
    output_dir: Optional[Path] = None,
    seed: int = 42,
    live: bool = False,
) -> ScenarioReasoningEvalReport:
    """
    Full pipeline eval (live/GPU). Blocked unless execution-sprint gate is set.
    """
    assert_execution_sprint_allowed(full_pipeline=True)
    batch_runner = ScenarioReasoningBatch(dry_run=False, live=live, seed=seed)
    full_report = batch_runner.run_all_fixtures()
    per_theta_slice: List[Dict[str, Any]] = []

    for batch in full_report.get("enumerated_batches", []):
        per_theta_slice.append(
            {
                "path_mode": "enumerated",
                "scenario_type": "enterprise",
                "theta": {},
                "metrics": _metrics_for_batch(batch),
                "path_count": batch["path_count"],
                "fixture_id": batch.get("fixture_id", ""),
            }
        )
    for batch in full_report.get("exploratory_batches", []):
        per_theta_slice.append(
            {
                "path_mode": "exploratory",
                "scenario_type": "causal",
                "theta": {},
                "metrics": _metrics_for_batch(batch),
                "path_count": batch["path_count"],
                "fixture_id": batch.get("fixture_id", ""),
            }
        )

    goal_eval = evaluate_goal_preservation()
    coherence_eval = evaluate_coherence()
    return build_reasoning_eval_report(
        per_theta_slice=per_theta_slice,
        goal_preservation=goal_eval,
        coherence=coherence_eval,
        smoke_mode=False,
        execution_sprint=True,
        provider_mode=effective_provider_mode(live_requested=live),
    )
