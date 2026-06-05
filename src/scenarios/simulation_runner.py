"""
Bundled scenario simulation runner (S5-02).

Simulation path: θ → world → trace (mock/smoke by default).
Supports two path modes:
  * ``wide`` — parameter grids, tree expansion, Monte Carlo (many scenarios)
  * ``bounded`` — fixed stages (good/bad/worst or N demo stages)
"""

from __future__ import annotations

import json
import uuid
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional

from src.risk.enterprise_theta import EnterpriseRiskTheta
from src.scenarios.causal.generator import CausalScenarioGenerator
from src.scenarios.causal.taxonomy import CausalTheta
from src.scenarios.resource_gate import assert_mock_or_gated, effective_provider_mode
from src.scenarios.theta_mapping import enterprise_theta_to_causal_slice
from src.search.cards import ActionCard, AlgorithmCard, NodeCard, SearchOperator
from src.search.game_theta import GameTheoreticTheta
from src.search.graph_monitor import SearchGraphMonitor

_REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_FIXTURES_PATH = _REPO_ROOT / "data" / "eval" / "simulation_fixtures.json"
GAME_BOUNDED_FIXTURE_PATH = _REPO_ROOT / "data" / "scenarios" / "game_bounded_default.json"


class ScenarioType(str, Enum):
    ENTERPRISE = "enterprise"
    CAUSAL = "causal"


class PathMode(str, Enum):
    WIDE = "wide"
    BOUNDED = "bounded"


@dataclass
class SimulationFixture:
    fixture_id: str
    scenario_type: ScenarioType
    path_mode: PathMode
    goal: str
    theta: Optional[Dict[str, Any]] = None
    theta_grid: Optional[List[Dict[str, Any]]] = None
    theta_samples: Optional[List[Dict[str, Any]]] = None
    stages: Optional[List[str]] = None
    expected_spans: List[str] = field(default_factory=list)
    profile: Optional[str] = None

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "SimulationFixture":
        return cls(
            fixture_id=data["fixture_id"],
            scenario_type=ScenarioType(data["scenario_type"]),
            path_mode=PathMode(data["path_mode"]),
            goal=data.get("goal", ""),
            theta=data.get("theta"),
            theta_grid=data.get("theta_grid"),
            theta_samples=data.get("theta_samples"),
            stages=data.get("stages"),
            expected_spans=list(data.get("expected_spans", [])),
            profile=data.get("profile"),
        )

    @property
    def is_game_bounded(self) -> bool:
        return (
            self.profile == "game_bounded"
            or self.fixture_id == "game_bounded_default"
        )


def load_game_bounded_default_fixture() -> SimulationFixture:
    """Load bounded game-θ smoke fixture (S6-07)."""
    if not GAME_BOUNDED_FIXTURE_PATH.is_file():
        raise FileNotFoundError(f"Game bounded fixture not found: {GAME_BOUNDED_FIXTURE_PATH}")
    data = json.loads(GAME_BOUNDED_FIXTURE_PATH.read_text(encoding="utf-8"))
    if "profile" not in data:
        data = {**data, "profile": "game_bounded"}
    return SimulationFixture.from_dict(data)


def load_simulation_fixtures(path: Path | str | None = None) -> List[SimulationFixture]:
    fixture_path = Path(path) if path else DEFAULT_FIXTURES_PATH
    if not fixture_path.is_file():
        raise FileNotFoundError(f"Simulation fixtures not found: {fixture_path}")
    payload = json.loads(fixture_path.read_text(encoding="utf-8"))
    fixtures = [SimulationFixture.from_dict(row) for row in payload.get("fixtures", [])]
    game_path = GAME_BOUNDED_FIXTURE_PATH
    if game_path.is_file() and not any(f.fixture_id == "game_bounded_default" for f in fixtures):
        fixtures.append(load_game_bounded_default_fixture())
    return fixtures


def _mock_span(name: str, trace_id: str, outputs: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    return {
        "name": name,
        "trace_id": trace_id,
        "span_id": str(uuid.uuid4()),
        "status": "ok",
        "outputs": outputs or {},
    }


def _run_enterprise_bounded(
    fixture: SimulationFixture,
    *,
    trace_id: str,
    provider_mode: str,
) -> Dict[str, Any]:
    theta_dict = fixture.theta or {}
    theta = EnterpriseRiskTheta(**{k: v for k, v in theta_dict.items() if k != "focus_sections"})
    stages = fixture.stages or ["stage_1", "stage_2", "stage_3"]
    spans = [_mock_span(s, trace_id, {"stage": s}) for s in fixture.expected_spans]
    stage_results = [
        {
            "stage": stage,
            "theta": theta.to_dict(),
            "causal_slice": enterprise_theta_to_causal_slice(theta).to_dict(),
            "scenario_count": min(theta.num_scenarios, len(stages)),
        }
        for stage in stages[: theta.num_scenarios]
    ]
    return {
        "scenario_type": ScenarioType.ENTERPRISE.value,
        "path_mode": PathMode.BOUNDED.value,
        "trace_id": trace_id,
        "provider_mode": provider_mode,
        "goal": fixture.goal,
        "theta": theta.to_dict(),
        "stages": stage_results,
        "spans": spans,
        "scenario_count": len(stage_results),
    }


def _run_enterprise_wide(
    fixture: SimulationFixture,
    *,
    trace_id: str,
    provider_mode: str,
) -> Dict[str, Any]:
    grid = fixture.theta_grid or [fixture.theta or {}]
    paths = []
    for i, row in enumerate(grid):
        theta = EnterpriseRiskTheta(**{k: v for k, v in row.items() if k != "focus_sections"})
        paths.append(
            {
                "path_index": i,
                "theta": theta.to_dict(),
                "causal_slice": enterprise_theta_to_causal_slice(theta).to_dict(),
                "mock_scenarios": theta.num_scenarios,
            }
        )
    spans = [_mock_span(n, trace_id) for n in fixture.expected_spans]
    return {
        "scenario_type": ScenarioType.ENTERPRISE.value,
        "path_mode": PathMode.WIDE.value,
        "trace_id": trace_id,
        "provider_mode": provider_mode,
        "goal": fixture.goal,
        "path_count": len(paths),
        "paths": paths,
        "spans": spans,
    }


def _run_causal_bounded(
    fixture: SimulationFixture,
    *,
    trace_id: str,
    provider_mode: str,
    seed: int,
) -> Dict[str, Any]:
    theta_dict = fixture.theta or {}
    theta = CausalTheta(**theta_dict)
    gen = CausalScenarioGenerator(seed=seed)
    stages = fixture.stages or ["good", "bad", "worst"]
    stage_results = []
    for stage in stages:
        inst = gen.instantiate(theta)
        stage_results.append(
            {
                "stage": stage,
                "prompt_preview": inst.prompt[:120],
                "answer": inst.answer,
                "metadata": inst.metadata,
            }
        )
    spans = [_mock_span(n, trace_id) for n in fixture.expected_spans]
    return {
        "scenario_type": ScenarioType.CAUSAL.value,
        "path_mode": PathMode.BOUNDED.value,
        "trace_id": trace_id,
        "provider_mode": provider_mode,
        "goal": fixture.goal,
        "theta": theta.to_dict(),
        "stages": stage_results,
        "spans": spans,
        "scenario_count": len(stage_results),
    }


def _run_game_bounded(
    fixture: SimulationFixture,
    *,
    trace_id: str,
    provider_mode: str,
) -> Dict[str, Any]:
    """Bounded game-θ path with SearchGraphMonitor visit/expansion/prune events."""
    theta = GameTheoreticTheta.from_dict(fixture.theta or {})
    stages = fixture.stages or [f"stage_{i}" for i in range(theta.num_stages)]
    monitor = SearchGraphMonitor()
    algo = AlgorithmCard.new("bounded_rollout", SearchOperator.ROLLOUT, {"budget": theta.num_stages})
    root = NodeCard.new("game_root", depth=0, theta_slice=theta.to_dict())
    monitor.record_visit(root, algo.algorithm_id, span_id=trace_id)

    stage_results: List[Dict[str, Any]] = []
    current = root
    for stage_idx, stage_name in enumerate(stages[: theta.num_stages]):
        monitor.record_expansion(current, algo.algorithm_id, span_id=trace_id)
        action_idx = theta.action_index_at_stage(stage_idx)
        vector_slice = theta.stage_slice(stage_idx)
        action = ActionCard.new(
            node_id=current.node_id,
            stage=stage_idx,
            vector_slice=vector_slice,
            label=f"action_{action_idx}",
            discrete_index=action_idx,
        )
        child = NodeCard.new(
            f"{stage_name}:{action_idx}",
            depth=current.depth + 1,
            parent_id=current.node_id,
            theta_slice={"stage": stage_idx, "action_index": action_idx},
        )
        monitor.record_visit(child, algo.algorithm_id, span_id=trace_id)
        if action_idx == 0 and stage_idx > 0:
            monitor.record_prune(child, "dominated_action", algo.algorithm_id)
        stage_results.append(
            {
                "stage": stage_name,
                "action_index": action_idx,
                "vector_slice": vector_slice,
                "action": action.to_dict(),
            }
        )
        current = child

    graph_report = monitor.snapshot()
    spans = [_mock_span(n, trace_id) for n in fixture.expected_spans]
    return {
        "scenario_type": "game",
        "path_mode": PathMode.BOUNDED.value,
        "profile": "game_bounded",
        "trace_id": trace_id,
        "provider_mode": provider_mode,
        "goal": fixture.goal,
        "theta": theta.to_dict(),
        "stages": stage_results,
        "spans": spans,
        "scenario_count": len(stage_results),
        "search_graph": graph_report.to_dict(),
    }


def _run_causal_wide(
    fixture: SimulationFixture,
    *,
    trace_id: str,
    provider_mode: str,
    seed: int,
) -> Dict[str, Any]:
    samples = fixture.theta_samples or fixture.theta_grid or [fixture.theta or {}]
    gen = CausalScenarioGenerator(seed=seed)
    paths = []
    for i, row in enumerate(samples):
        theta = CausalTheta(**row)
        inst = gen.instantiate(theta)
        paths.append(
            {
                "path_index": i,
                "theta": theta.to_dict(),
                "answer": inst.answer,
                "intervention_type": theta.intervention_type,
            }
        )
    spans = [_mock_span(n, trace_id) for n in fixture.expected_spans]
    return {
        "scenario_type": ScenarioType.CAUSAL.value,
        "path_mode": PathMode.WIDE.value,
        "trace_id": trace_id,
        "provider_mode": provider_mode,
        "goal": fixture.goal,
        "path_count": len(paths),
        "paths": paths,
        "spans": spans,
    }


@dataclass
class ScenarioSimulationRunner:
    """
    Runs bundled simulation fixtures in mock/smoke mode.

    Args:
        fixtures_path: Path to ``simulation_fixtures.json``.
        seed: RNG seed for causal template generation.
        dry_run: When True (default), never invoke live providers.
        live: Request live provider (requires resource gate).
    """

    fixtures_path: Optional[Path] = None
    seed: int = 42
    dry_run: bool = True
    live: bool = False

    def run_fixture(self, fixture: SimulationFixture) -> Dict[str, Any]:
        assert_mock_or_gated(live_requested=self.live)
        provider_mode = effective_provider_mode(live_requested=self.live)
        if self.dry_run:
            provider_mode = "mock"

        trace_id = str(uuid.uuid4())
        if fixture.is_game_bounded:
            return _run_game_bounded(fixture, trace_id=trace_id, provider_mode=provider_mode)

        if fixture.scenario_type == ScenarioType.ENTERPRISE:
            if fixture.path_mode == PathMode.BOUNDED:
                return _run_enterprise_bounded(fixture, trace_id=trace_id, provider_mode=provider_mode)
            return _run_enterprise_wide(fixture, trace_id=trace_id, provider_mode=provider_mode)

        if fixture.path_mode == PathMode.BOUNDED:
            return _run_causal_bounded(
                fixture, trace_id=trace_id, provider_mode=provider_mode, seed=self.seed
            )
        return _run_causal_wide(
            fixture, trace_id=trace_id, provider_mode=provider_mode, seed=self.seed
        )

    def run_all(
        self,
        *,
        scenario_type: Optional[str] = None,
        fixture_ids: Optional[List[str]] = None,
    ) -> Dict[str, Any]:
        fixtures = load_simulation_fixtures(self.fixtures_path)
        if scenario_type:
            fixtures = [
                f
                for f in fixtures
                if f.scenario_type.value == scenario_type
                and not (scenario_type == "causal" and f.is_game_bounded)
            ]
        if fixture_ids:
            ids = set(fixture_ids)
            fixtures = [f for f in fixtures if f.fixture_id in ids]

        results = [self.run_fixture(f) for f in fixtures]
        return {
            "schema": "scenario_simulation_run_v1",
            "dry_run": self.dry_run,
            "provider_mode": effective_provider_mode(live_requested=self.live),
            "fixture_count": len(results),
            "results": results,
        }
