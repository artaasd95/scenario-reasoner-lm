"""
Batch reasoning over scenario paths with mock/smoke provider (S6-03 / S7-03).

Runs causal/DSPy-style reasoning per path in a batch. Each path carries
scenario goal and θ constraints into the reasoning loop.
"""

from __future__ import annotations

import uuid
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Union

from src.metrics.goal_preservation_metrics import score_goal_alignment
from src.scenarios.enumerated_path_generator import ScenarioPath, generate_from_fixture as gen_enum_fixture
from src.scenarios.enumerated_path_generator import load_enumerated_fixtures
from src.scenarios.exploratory_path_generator import ExploratoryPath, generate_from_exploratory_fixture
from src.scenarios.exploratory_path_generator import load_exploratory_fixtures
from src.scenarios.resource_gate import assert_mock_or_gated, effective_provider_mode
from src.scenarios.simulation_runner import _mock_span

# Offline reasoning stubs keyed by scenario type and stage severity rank
_ENTERPRISE_REASONING: Dict[int, str] = {
    0: "Risk Factors baseline: Taiwan supply concentration noted in 10-K. Step 1: monitor foundry lead times.",
    1: "Elevated: export controls tighten on advanced packaging. Step 2: OEM penalty clauses become likely.",
    2: "Severe: sole-foundry cutoff halts controller production for automotive safety lines.",
    3: "Critical: major OEMs invoke penalty clauses and switch to competitors from acme 10-K disclosures.",
    4: "Catastrophic: five ranked catastrophic scenarios from Risk Factors supply chain cascade. Therefore worst-case OEM exodus.",
}

_CAUSAL_REASONING: Dict[str, str] = {
    "good": "Step 1: Heat causes expansion. Step 2: Expansion causes stress. Therefore the final outcome is stable.",
    "neutral": "Step 1: A leads to B. Step 2: B leads to C. Therefore moderate outcome under direct chain.",
    "bad": "Step 1: A leads to B. Step 2: B leads to C. Therefore adverse outcome without intervention.",
    "worst": "Step 1: A leads to B. Step 2: B leads to C. Therefore catastrophic final outcome.",
}


@dataclass
class PathReasoningResult:
    path_id: str
    stage: str
    scenario_goal: str
    theta: Dict[str, Any]
    causal_slice: Dict[str, Any]
    reasoning_text: str
    goal_scores: Dict[str, float]
    trace_id: str
    spans: List[Dict[str, Any]] = field(default_factory=list)
    provider_mode: str = "mock"
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


def _mock_reasoning_for_path(path: Union[ScenarioPath, ExploratoryPath, Dict[str, Any]]) -> str:
    """Generate mock reasoning text that respects goal and θ constraints."""
    if isinstance(path, ScenarioPath):
        rank = path.metadata.get("severity_rank", path.stage_index)
        if path.scenario_type == "enterprise":
            text = _ENTERPRISE_REASONING.get(rank, _ENTERPRISE_REASONING[4])
            return f"{text} Goal: {path.scenario_goal}. filing_id={path.theta.get('filing_id', '')}."
        return _CAUSAL_REASONING.get(path.stage, _CAUSAL_REASONING["neutral"])

    if isinstance(path, ExploratoryPath):
        if path.scenario_type == "enterprise":
            floor = path.theta.get("severity_floor", "high")
            return (
                f"Exploratory enterprise path: Risk Factors 10-K supply chain at severity_floor={floor}. "
                f"Therefore ranked scenarios for {path.scenario_goal}."
            )
        answer = path.metadata.get("answer", "outcome")
        return (
            f"Step 1: A leads to B. Step 2: B leads to C. "
            f"Therefore {answer}. intervention_type={path.theta.get('intervention_type', 'direct')}."
        )

    row = path
    st = row.get("scenario_type", "enterprise")
    if st == "enterprise":
        return _ENTERPRISE_REASONING.get(0, "") + f" Goal: {row.get('scenario_goal', '')}."
    return _CAUSAL_REASONING.get("neutral", "")


def _path_identity(path: Union[ScenarioPath, ExploratoryPath, Dict[str, Any]]) -> tuple:
    if isinstance(path, ScenarioPath):
        return (path.path_id, path.stage, path.scenario_goal, path.theta, path.causal_slice, path.metadata)
    if isinstance(path, ExploratoryPath):
        return (
            f"exploratory_{path.path_index}",
            str(path.path_index),
            path.scenario_goal,
            path.theta,
            path.causal_slice,
            path.metadata,
        )
    return (
        path.get("path_id", "unknown"),
        path.get("stage", ""),
        path.get("scenario_goal", ""),
        path.get("theta", {}),
        path.get("causal_slice", {}),
        path.get("metadata", {}),
    )


def run_reasoning_for_path(
    path: Union[ScenarioPath, ExploratoryPath, Dict[str, Any]],
    *,
    trace_id: Optional[str] = None,
    provider_mode: str = "mock",
) -> PathReasoningResult:
    """Run mock reasoning for one path; θ constraints feed goal alignment scoring."""
    path_id, stage, goal, theta, causal_slice, metadata = _path_identity(path)
    tid = trace_id or str(uuid.uuid4())
    if isinstance(path, dict) and path.get("reasoning_text"):
        reasoning = path["reasoning_text"]
    else:
        reasoning = _mock_reasoning_for_path(path)
    scores = score_goal_alignment(reasoning, goal, theta)
    spans = [
        _mock_span("prompt", tid, {"theta": theta, "goal": goal}),
        _mock_span("reasoning", tid, {"text_preview": reasoning[:200]}),
        _mock_span("answer", tid, {"on_target_composite": scores["on_target_composite"]}),
    ]
    return PathReasoningResult(
        path_id=path_id,
        stage=stage,
        scenario_goal=goal,
        theta=theta,
        causal_slice=causal_slice if isinstance(causal_slice, dict) else {},
        reasoning_text=reasoning,
        goal_scores=scores,
        trace_id=tid,
        spans=spans,
        provider_mode=provider_mode,
        metadata=dict(metadata) if isinstance(metadata, dict) else {},
    )


@dataclass
class ScenarioReasoningBatch:
    """
    Batch dispatch over scenario paths using mock/smoke provider by default.

    Args:
        dry_run: When True (default), mock provider only.
        live: Request live provider (requires resource gate).
        seed: RNG seed for path generators.
    """

    dry_run: bool = True
    live: bool = False
    seed: int = 42

    def run_paths(
        self,
        paths: Sequence[Union[ScenarioPath, ExploratoryPath, Dict[str, Any]]],
    ) -> Dict[str, Any]:
        assert_mock_or_gated(live_requested=self.live)
        provider_mode = effective_provider_mode(live_requested=self.live)
        if self.dry_run:
            provider_mode = "mock"

        batch_trace = str(uuid.uuid4())
        results = [
            run_reasoning_for_path(p, trace_id=batch_trace, provider_mode=provider_mode).to_dict()
            for p in paths
        ]
        on_target = sum(1 for r in results if r["goal_scores"]["on_target_composite"] >= 0.55)
        return {
            "schema": "scenario_reasoning_batch_v1",
            "dry_run": self.dry_run,
            "provider_mode": provider_mode,
            "batch_trace_id": batch_trace,
            "path_count": len(results),
            "on_target_count": on_target,
            "results": results,
        }

    def run_enumerated_fixture(self, fixture_id: str) -> Dict[str, Any]:
        fixtures = load_enumerated_fixtures()
        fx = next(f for f in fixtures if f.fixture_id == fixture_id)
        paths = gen_enum_fixture(fx, seed=self.seed)
        report = self.run_paths(paths)
        report["fixture_id"] = fixture_id
        report["path_mode"] = "enumerated"
        return report

    def run_exploratory_fixture(self, fixture_id: str) -> Dict[str, Any]:
        fixtures = load_exploratory_fixtures()
        fx = next(f for f in fixtures if f.fixture_id == fixture_id)
        paths = generate_from_exploratory_fixture(fx)
        report = self.run_paths(paths)
        report["fixture_id"] = fixture_id
        report["path_mode"] = "exploratory"
        return report

    def run_all_fixtures(self) -> Dict[str, Any]:
        enum_reports = [
            self.run_enumerated_fixture(f.fixture_id) for f in load_enumerated_fixtures()
        ]
        expl_reports = [
            self.run_exploratory_fixture(f.fixture_id) for f in load_exploratory_fixtures()
        ]
        return {
            "schema": "scenario_reasoning_batch_run_v1",
            "dry_run": self.dry_run,
            "provider_mode": effective_provider_mode(live_requested=self.live),
            "enumerated_batches": enum_reports,
            "exploratory_batches": expl_reports,
            "batch_count": len(enum_reports) + len(expl_reports),
        }
