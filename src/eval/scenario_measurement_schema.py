"""
Stable JSON schema for multi-scenario measurement harness (S5-04).

Extends robustness_eval ``per_theta`` / ``aggregate`` shape and enterprise eval
report fields for θ-stratified reporting.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional


SCHEMA_VERSION = "1.0.0"


@dataclass
class MeasurementRunMetadata:
    smoke_mode: bool = True
    offline: bool = True
    provider_mode: str = "mock"
    eval_set_paths: List[str] = field(default_factory=list)
    run_id: str = ""
    created_at: str = field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )


@dataclass
class ThetaSliceScore:
    scenario_type: str
    path_mode: str
    theta: Dict[str, Any]
    metrics: Dict[str, float] = field(default_factory=dict)
    n_evaluated: int = 0
    fixture_id: str = ""


@dataclass
class ScenarioMeasurementReport:
    schema_version: str = SCHEMA_VERSION
    metadata: MeasurementRunMetadata = field(default_factory=MeasurementRunMetadata)
    per_scenario_type: Dict[str, Dict[str, float]] = field(default_factory=dict)
    per_theta_slice: List[Dict[str, Any]] = field(default_factory=list)
    aggregate: Dict[str, float] = field(default_factory=dict)
    enterprise_eval_ref: Optional[Dict[str, Any]] = None

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    def write_json(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(self.to_dict(), indent=2), encoding="utf-8")


def build_measurement_report(
    *,
    per_theta_slice: List[Dict[str, Any]],
    enterprise_eval: Optional[Dict[str, Any]] = None,
    goal_preservation: Optional[Dict[str, Any]] = None,
    smoke_mode: bool = True,
    provider_mode: str = "mock",
) -> ScenarioMeasurementReport:
    """Assemble report from simulation/metric sub-harness outputs."""
    by_type: Dict[str, List[Dict[str, float]]] = {}
    for row in per_theta_slice:
        st = row.get("scenario_type", "unknown")
        metrics = row.get("metrics", {})
        by_type.setdefault(st, []).append(metrics)

    per_scenario_type: Dict[str, Dict[str, float]] = {}
    for st, metric_rows in by_type.items():
        keys = {k for m in metric_rows for k in m}
        per_scenario_type[st] = {
            k: round(
                sum(m.get(k, 0.0) for m in metric_rows) / max(len(metric_rows), 1),
                4,
            )
            for k in sorted(keys)
        }

    aggregate: Dict[str, float] = {}
    if goal_preservation:
        aggregate.update(goal_preservation.get("aggregate", {}))
    if enterprise_eval:
        aggregate.update(enterprise_eval.get("aggregate_scores", {}))

    all_metrics: Dict[str, List[float]] = {}
    for row in per_theta_slice:
        for name, val in row.get("metrics", {}).items():
            if isinstance(val, (int, float)):
                all_metrics.setdefault(name, []).append(float(val))
    for name, vals in all_metrics.items():
        if name not in aggregate:
            aggregate[name] = round(sum(vals) / len(vals), 4)

    meta = MeasurementRunMetadata(
        smoke_mode=smoke_mode,
        offline=True,
        provider_mode=provider_mode,
        eval_set_paths=[
            "data/eval/simulation_fixtures.json",
            "data/eval/goal_preservation_fixtures.jsonl",
        ],
    )

    return ScenarioMeasurementReport(
        metadata=meta,
        per_scenario_type=per_scenario_type,
        per_theta_slice=per_theta_slice,
        aggregate=aggregate,
        enterprise_eval_ref=enterprise_eval,
    )
