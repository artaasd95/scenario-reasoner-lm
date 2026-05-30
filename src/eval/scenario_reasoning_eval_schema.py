"""
Scenario-set reasoning eval schema (S6-05 / S7-05).

Extends scenario_measurement_schema for exploratory vs enumerated path types
with θ-stratified reporting.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

from src.eval.scenario_measurement_schema import SCHEMA_VERSION

REASONING_EVAL_SCHEMA_VERSION = "1.0.0"


@dataclass
class ReasoningEvalRunMetadata:
    smoke_mode: bool = True
    execution_sprint: bool = False
    provider_mode: str = "mock"
    eval_set_paths: List[str] = field(default_factory=list)
    run_id: str = ""
    created_at: str = field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )


@dataclass
class PathTypeScore:
    path_mode: str  # "enumerated" | "exploratory"
    scenario_type: str
    theta: Dict[str, Any]
    metrics: Dict[str, float] = field(default_factory=dict)
    path_count: int = 0
    fixture_id: str = ""


@dataclass
class ScenarioReasoningEvalReport:
    schema_version: str = REASONING_EVAL_SCHEMA_VERSION
    measurement_schema_ref: str = SCHEMA_VERSION
    metadata: ReasoningEvalRunMetadata = field(default_factory=ReasoningEvalRunMetadata)
    per_path_type: Dict[str, Dict[str, float]] = field(default_factory=dict)
    per_theta_slice: List[Dict[str, Any]] = field(default_factory=list)
    aggregate: Dict[str, float] = field(default_factory=dict)
    goal_preservation_ref: Optional[Dict[str, Any]] = None
    coherence_ref: Optional[Dict[str, Any]] = None

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    def write_json(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(self.to_dict(), indent=2), encoding="utf-8")


def build_reasoning_eval_report(
    *,
    per_theta_slice: List[Dict[str, Any]],
    goal_preservation: Optional[Dict[str, Any]] = None,
    coherence: Optional[Dict[str, Any]] = None,
    smoke_mode: bool = True,
    execution_sprint: bool = False,
    provider_mode: str = "mock",
) -> ScenarioReasoningEvalReport:
    """Assemble eval report from batch reasoning and metric sub-harness outputs."""
    by_path_type: Dict[str, List[Dict[str, float]]] = {}
    for row in per_theta_slice:
        pm = row.get("path_mode", "unknown")
        metrics = row.get("metrics", {})
        by_path_type.setdefault(pm, []).append(metrics)

    per_path_type: Dict[str, Dict[str, float]] = {}
    for pm, metric_rows in by_path_type.items():
        keys = {k for m in metric_rows for k in m}
        per_path_type[pm] = {
            k: round(
                sum(m.get(k, 0.0) for m in metric_rows) / max(len(metric_rows), 1),
                4,
            )
            for k in sorted(keys)
        }

    aggregate: Dict[str, float] = {}
    if goal_preservation:
        aggregate.update(goal_preservation.get("aggregate", {}))
    if coherence:
        aggregate.update(coherence.get("aggregate", {}))

    all_metrics: Dict[str, List[float]] = {}
    for row in per_theta_slice:
        for name, val in row.get("metrics", {}).items():
            if isinstance(val, (int, float)):
                all_metrics.setdefault(name, []).append(float(val))
    for name, vals in all_metrics.items():
        if name not in aggregate:
            aggregate[name] = round(sum(vals) / len(vals), 4)

    meta = ReasoningEvalRunMetadata(
        smoke_mode=smoke_mode,
        execution_sprint=execution_sprint,
        provider_mode=provider_mode,
        eval_set_paths=[
            "data/scenarios/enumerated_path_fixtures.json",
            "data/scenarios/exploratory_path_fixtures.json",
            "data/eval/cross_scenario_coherence_fixtures.jsonl",
            "data/eval/goal_preservation_fixtures.jsonl",
        ],
    )

    return ScenarioReasoningEvalReport(
        metadata=meta,
        per_path_type=per_path_type,
        per_theta_slice=per_theta_slice,
        aggregate=aggregate,
        goal_preservation_ref=goal_preservation,
        coherence_ref=coherence,
    )
