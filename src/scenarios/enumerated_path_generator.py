"""
Enumerated scenario path generator (S6-02 / S7-02).

Emits a fixed small set of scenario paths (good/bad/worst or N-stage ladders)
from θ anchors. Paths map to ``EnterpriseRiskTheta`` / causal θ slices used by
the enterprise demo.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Union

from src.risk.enterprise_theta import EnterpriseRiskTheta
from src.scenarios.causal.generator import CausalScenarioGenerator
from src.scenarios.causal.taxonomy import CausalTheta
from src.scenarios.theta_mapping import enterprise_theta_to_causal_slice

_REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_FIXTURES_PATH = _REPO_ROOT / "data" / "scenarios" / "enumerated_path_fixtures.json"

DEFAULT_ENTERPRISE_STAGES = (
    "baseline",
    "elevated",
    "severe",
    "critical",
    "catastrophic",
)
DEFAULT_CAUSAL_STAGES = ("good", "neutral", "bad", "worst")
DEFAULT_SEVERITY_LADDER = ("good", "bad", "worst")


class ScenarioPathKind(str, Enum):
    ENTERPRISE = "enterprise"
    CAUSAL = "causal"


@dataclass
class ScenarioPath:
    """One bounded scenario path with θ anchor, stage label, and causal slice."""

    path_id: str
    stage: str
    stage_index: int
    scenario_goal: str
    theta: Dict[str, Any]
    causal_slice: Dict[str, Any]
    scenario_type: str
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class EnumeratedPathFixture:
    fixture_id: str
    scenario_type: ScenarioPathKind
    goal: str
    theta: Dict[str, Any]
    stages: List[str]
    expected_path_count: int

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "EnumeratedPathFixture":
        return cls(
            fixture_id=data["fixture_id"],
            scenario_type=ScenarioPathKind(data["scenario_type"]),
            goal=data["goal"],
            theta=dict(data["theta"]),
            stages=list(data["stages"]),
            expected_path_count=int(data.get("expected_path_count", len(data["stages"]))),
        )


def load_enumerated_fixtures(path: Path | str | None = None) -> List[EnumeratedPathFixture]:
    fixture_path = Path(path) if path else DEFAULT_FIXTURES_PATH
    if not fixture_path.is_file():
        raise FileNotFoundError(f"Enumerated path fixtures not found: {fixture_path}")
    payload = json.loads(fixture_path.read_text(encoding="utf-8"))
    return [EnumeratedPathFixture.from_dict(row) for row in payload.get("fixtures", [])]


def _resolve_stages(
    stages: Optional[Sequence[str]],
    *,
    num_scenarios: int,
    scenario_type: ScenarioPathKind,
) -> List[str]:
    if stages:
        return list(stages)
    if scenario_type == ScenarioPathKind.ENTERPRISE:
        return list(DEFAULT_ENTERPRISE_STAGES[:num_scenarios])
    return list(DEFAULT_CAUSAL_STAGES[: max(num_scenarios, 3)])


def generate_enumerated_paths(
    theta: Union[EnterpriseRiskTheta, CausalTheta, Dict[str, Any]],
    *,
    scenario_goal: str = "",
    stages: Optional[Sequence[str]] = None,
    scenario_type: Optional[ScenarioPathKind] = None,
    seed: int = 42,
    path_prefix: str = "path",
) -> List[ScenarioPath]:
    """
    Generate a fixed ordered set of scenario paths from a θ anchor.

    Enterprise paths attach ``enterprise_theta_to_causal_slice`` for cross-harness
    reporting. Causal paths instantiate template previews per stage.
    """
    if isinstance(theta, dict):
        if scenario_type is None:
            scenario_type = (
                ScenarioPathKind.ENTERPRISE
                if "filing_id" in theta
                else ScenarioPathKind.CAUSAL
            )
        if scenario_type == ScenarioPathKind.ENTERPRISE:
            theta_obj: Union[EnterpriseRiskTheta, CausalTheta] = EnterpriseRiskTheta(
                **{k: v for k, v in theta.items() if k != "focus_sections"}
            )
        else:
            theta_obj = CausalTheta(**theta)
    else:
        theta_obj = theta
        if scenario_type is None:
            scenario_type = (
                ScenarioPathKind.ENTERPRISE
                if isinstance(theta_obj, EnterpriseRiskTheta)
                else ScenarioPathKind.CAUSAL
            )

    if scenario_type == ScenarioPathKind.ENTERPRISE:
        assert isinstance(theta_obj, EnterpriseRiskTheta)
        stage_list = _resolve_stages(
            stages, num_scenarios=theta_obj.num_scenarios, scenario_type=scenario_type
        )[: theta_obj.num_scenarios]
        causal = enterprise_theta_to_causal_slice(theta_obj)
        paths: List[ScenarioPath] = []
        for i, stage in enumerate(stage_list):
            paths.append(
                ScenarioPath(
                    path_id=f"{path_prefix}_{i}",
                    stage=stage,
                    stage_index=i,
                    scenario_goal=scenario_goal,
                    theta=theta_obj.to_dict(),
                    causal_slice=causal.to_dict(),
                    scenario_type=scenario_type.value,
                    metadata={"severity_rank": i, "scenario_count": theta_obj.num_scenarios},
                )
            )
        return paths

    assert isinstance(theta_obj, CausalTheta)
    stage_list = _resolve_stages(
        stages, num_scenarios=len(stages or DEFAULT_CAUSAL_STAGES), scenario_type=scenario_type
    )
    gen = CausalScenarioGenerator(seed=seed)
    paths = []
    for i, stage in enumerate(stage_list):
        inst = gen.instantiate(theta_obj)
        paths.append(
            ScenarioPath(
                path_id=f"{path_prefix}_{i}",
                stage=stage,
                stage_index=i,
                scenario_goal=scenario_goal,
                theta=theta_obj.to_dict(),
                causal_slice=theta_obj.to_dict(),
                scenario_type=scenario_type.value,
                metadata={
                    "severity_rank": i,
                    "prompt_preview": inst.prompt[:120],
                    "answer": inst.answer,
                },
            )
        )
    return paths


def generate_from_fixture(
    fixture: EnumeratedPathFixture,
    *,
    seed: int = 42,
) -> List[ScenarioPath]:
    """Round-trip helper: fixture → paths with stable ordering."""
    return generate_enumerated_paths(
        fixture.theta,
        scenario_goal=fixture.goal,
        stages=fixture.stages,
        scenario_type=fixture.scenario_type,
        seed=seed,
        path_prefix=fixture.fixture_id,
    )


def paths_to_dict(paths: Sequence[ScenarioPath]) -> Dict[str, Any]:
    """Serialize path set for fixtures and eval harnesses."""
    return {
        "schema": "enumerated_scenario_paths_v1",
        "path_count": len(paths),
        "paths": [p.to_dict() for p in paths],
    }
