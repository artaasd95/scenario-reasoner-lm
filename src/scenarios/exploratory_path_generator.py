"""
Exploratory scenario path generator (S7-01).

Samples many scenario paths from θ perturbations (parameter sweep, tree
expansion, or Monte Carlo) with bounded fan-out config.
"""

from __future__ import annotations

import json
import random
from dataclasses import asdict, dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Union

from src.risk.enterprise_theta import EnterpriseRiskTheta, EnterpriseRiskThetaSampler
from src.scenarios.causal.generator import CausalScenarioGenerator
from src.scenarios.causal.taxonomy import CausalTheta, CausalThetaSampler
from src.scenarios.theta_mapping import enterprise_theta_to_causal_slice

_REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_FIXTURES_PATH = _REPO_ROOT / "data" / "scenarios" / "exploratory_path_fixtures.json"


class ExploratoryStrategy(str, Enum):
    GRID = "grid"
    MONTE_CARLO = "monte_carlo"
    TREE_EXPAND = "tree_expand"


@dataclass
class ExploratoryPathConfig:
    """Bounded fan-out config for wide θ search."""

    max_paths: int = 12
    seed: int = 42
    strategy: ExploratoryStrategy = ExploratoryStrategy.GRID

    def to_dict(self) -> Dict[str, Any]:
        return {
            "max_paths": self.max_paths,
            "seed": self.seed,
            "strategy": self.strategy.value,
        }


@dataclass
class ExploratoryPath:
    path_index: int
    theta: Dict[str, Any]
    causal_slice: Dict[str, Any]
    scenario_type: str
    scenario_goal: str = ""
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class ExploratoryPathFixture:
    fixture_id: str
    scenario_type: str
    goal: str
    config: ExploratoryPathConfig
    theta_anchor: Dict[str, Any]
    theta_grid: Optional[List[Dict[str, Any]]] = None
    theta_samples: Optional[List[Dict[str, Any]]] = None
    min_path_count: int = 3

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ExploratoryPathFixture":
        cfg_raw = data.get("config", {})
        return cls(
            fixture_id=data["fixture_id"],
            scenario_type=data["scenario_type"],
            goal=data["goal"],
            config=ExploratoryPathConfig(
                max_paths=int(cfg_raw.get("max_paths", 12)),
                seed=int(cfg_raw.get("seed", 42)),
                strategy=ExploratoryStrategy(cfg_raw.get("strategy", "grid")),
            ),
            theta_anchor=dict(data.get("theta_anchor", data.get("theta", {}))),
            theta_grid=data.get("theta_grid"),
            theta_samples=data.get("theta_samples"),
            min_path_count=int(data.get("min_path_count", 3)),
        )


def load_exploratory_fixtures(path: Path | str | None = None) -> List[ExploratoryPathFixture]:
    fixture_path = Path(path) if path else DEFAULT_FIXTURES_PATH
    if not fixture_path.is_file():
        raise FileNotFoundError(f"Exploratory path fixtures not found: {fixture_path}")
    payload = json.loads(fixture_path.read_text(encoding="utf-8"))
    return [ExploratoryPathFixture.from_dict(row) for row in payload.get("fixtures", [])]


def _cap_paths(paths: List[ExploratoryPath], max_paths: int) -> List[ExploratoryPath]:
    return paths[:max_paths]


def _enterprise_grid_paths(
    theta_rows: List[Dict[str, Any]],
    *,
    goal: str,
    max_paths: int,
) -> List[ExploratoryPath]:
    paths: List[ExploratoryPath] = []
    for i, row in enumerate(theta_rows):
        theta = EnterpriseRiskTheta(**{k: v for k, v in row.items() if k != "focus_sections"})
        paths.append(
            ExploratoryPath(
                path_index=i,
                theta=theta.to_dict(),
                causal_slice=enterprise_theta_to_causal_slice(theta).to_dict(),
                scenario_type="enterprise",
                scenario_goal=goal,
                metadata={"strategy": ExploratoryStrategy.GRID.value},
            )
        )
    return _cap_paths(paths, max_paths)


def _causal_grid_paths(
    theta_rows: List[Dict[str, Any]],
    *,
    goal: str,
    max_paths: int,
    seed: int,
) -> List[ExploratoryPath]:
    gen = CausalScenarioGenerator(seed=seed)
    paths: List[ExploratoryPath] = []
    for i, row in enumerate(theta_rows):
        theta = CausalTheta(**row)
        inst = gen.instantiate(theta)
        paths.append(
            ExploratoryPath(
                path_index=i,
                theta=theta.to_dict(),
                causal_slice=theta.to_dict(),
                scenario_type="causal",
                scenario_goal=goal,
                metadata={
                    "strategy": ExploratoryStrategy.GRID.value,
                    "answer": inst.answer,
                },
            )
        )
    return _cap_paths(paths, max_paths)


def _monte_carlo_enterprise(
    anchor: Dict[str, Any],
    *,
    goal: str,
    max_paths: int,
    seed: int,
) -> List[ExploratoryPath]:
    rng = random.Random(seed)
    base = EnterpriseRiskTheta(**{k: v for k, v in anchor.items() if k != "focus_sections"})
    floors = ["high", "catastrophic"]
    nums = list(range(max(1, base.num_scenarios - 1), min(base.num_scenarios + 2, 10)))
    paths: List[ExploratoryPath] = []
    for i in range(max_paths):
        n = rng.choice(nums)
        floor = rng.choice(floors)
        theta = EnterpriseRiskTheta(
            filing_id=base.filing_id,
            num_scenarios=n,
            severity_floor=floor,
            ranking_strategy=base.ranking_strategy,
        )
        paths.append(
            ExploratoryPath(
                path_index=i,
                theta=theta.to_dict(),
                causal_slice=enterprise_theta_to_causal_slice(theta).to_dict(),
                scenario_type="enterprise",
                scenario_goal=goal,
                metadata={"strategy": ExploratoryStrategy.MONTE_CARLO.value, "perturbation": i},
            )
        )
    return paths


def _monte_carlo_causal(
    anchor: Dict[str, Any],
    *,
    goal: str,
    max_paths: int,
    seed: int,
) -> List[ExploratoryPath]:
    rng = random.Random(seed)
    base = CausalTheta(**anchor)
    gen = CausalScenarioGenerator(seed=seed)
    paths: List[ExploratoryPath] = []
    for i in range(max_paths):
        chain = rng.randint(max(2, base.chain_length - 1), min(8, base.chain_length + 2))
        diff = rng.choice(["easy", "medium", "hard"])
        theta = CausalTheta(
            chain_length=chain,
            intervention_type=base.intervention_type,
            num_confounders=base.num_confounders,
            domain=base.domain,
            difficulty=diff,
            entity_count=min(base.entity_count + (i % 2), 5),
        )
        inst = gen.instantiate(theta)
        paths.append(
            ExploratoryPath(
                path_index=i,
                theta=theta.to_dict(),
                causal_slice=theta.to_dict(),
                scenario_type="causal",
                scenario_goal=goal,
                metadata={
                    "strategy": ExploratoryStrategy.MONTE_CARLO.value,
                    "answer": inst.answer,
                    "perturbation": i,
                },
            )
        )
    return paths


def _tree_expand_enterprise(
    anchor: Dict[str, Any],
    *,
    goal: str,
    max_paths: int,
    seed: int,
) -> List[ExploratoryPath]:
    """Simple breadth-first θ fan-out from anchor (bounded by max_paths)."""
    sampler = EnterpriseRiskThetaSampler(seed=seed)
    grid = sampler.grid(
        filing_ids=[anchor.get("filing_id", "acme_corp_10k")],
        num_scenarios_list=[3, 5, anchor.get("num_scenarios", 5)],
        severity_floors=["high", "catastrophic"],
    )
    rows = [g.to_dict() for g in grid]
    return _enterprise_grid_paths(rows, goal=goal, max_paths=max_paths)


def _tree_expand_causal(
    anchor: Dict[str, Any],
    *,
    goal: str,
    max_paths: int,
    seed: int,
) -> List[ExploratoryPath]:
    sampler = CausalThetaSampler(
        chain_length_range=(anchor.get("chain_length", 3), anchor.get("chain_length", 3) + 2),
        domains=[anchor.get("domain", "physical")],
        seed=seed,
    )
    grid = sampler.grid(
        chain_lengths=[anchor.get("chain_length", 3), anchor.get("chain_length", 3) + 1],
        intervention_types=[anchor.get("intervention_type", "direct")],
        domains=[anchor.get("domain", "physical")],
        difficulties=["easy", "medium"],
    )
    rows = [g.to_dict() for g in grid]
    return _causal_grid_paths(rows, goal=goal, max_paths=max_paths, seed=seed)


def generate_exploratory_paths(
    theta_anchor: Union[EnterpriseRiskTheta, CausalTheta, Dict[str, Any]],
    *,
    scenario_goal: str = "",
    config: Optional[ExploratoryPathConfig] = None,
    theta_grid: Optional[List[Dict[str, Any]]] = None,
    theta_samples: Optional[List[Dict[str, Any]]] = None,
    scenario_type: Optional[str] = None,
) -> List[ExploratoryPath]:
    """
    Sample many paths from θ perturbations with bounded fan-out.

    Strategies: ``grid`` (explicit rows or sampler grid), ``monte_carlo`` (RNG
    perturbations), ``tree_expand`` (BFS-style fan-out from anchor).
    """
    cfg = config or ExploratoryPathConfig()
    anchor = (
        theta_anchor.to_dict()
        if hasattr(theta_anchor, "to_dict")
        else dict(theta_anchor)
    )
    st = scenario_type or ("enterprise" if "filing_id" in anchor else "causal")

    if cfg.strategy == ExploratoryStrategy.MONTE_CARLO:
        if st == "enterprise":
            return _monte_carlo_enterprise(
                anchor, goal=scenario_goal, max_paths=cfg.max_paths, seed=cfg.seed
            )
        return _monte_carlo_causal(
            anchor, goal=scenario_goal, max_paths=cfg.max_paths, seed=cfg.seed
        )

    if cfg.strategy == ExploratoryStrategy.TREE_EXPAND:
        if st == "enterprise":
            return _tree_expand_enterprise(
                anchor, goal=scenario_goal, max_paths=cfg.max_paths, seed=cfg.seed
            )
        return _tree_expand_causal(
            anchor, goal=scenario_goal, max_paths=cfg.max_paths, seed=cfg.seed
        )

    rows = theta_grid or theta_samples
    if not rows and st == "enterprise":
        sampler = EnterpriseRiskThetaSampler(seed=cfg.seed)
        rows = [t.to_dict() for t in sampler.grid()]
    elif not rows:
        sampler = CausalThetaSampler(seed=cfg.seed)
        rows = [t.to_dict() for t in sampler.grid(chain_lengths=[3, 4], domains=["physical"])]

    if st == "enterprise":
        return _enterprise_grid_paths(rows or [anchor], goal=scenario_goal, max_paths=cfg.max_paths)
    return _causal_grid_paths(
        rows or [anchor], goal=scenario_goal, max_paths=cfg.max_paths, seed=cfg.seed
    )


def generate_from_exploratory_fixture(
    fixture: ExploratoryPathFixture,
) -> List[ExploratoryPath]:
    return generate_exploratory_paths(
        fixture.theta_anchor,
        scenario_goal=fixture.goal,
        config=fixture.config,
        theta_grid=fixture.theta_grid,
        theta_samples=fixture.theta_samples,
        scenario_type=fixture.scenario_type,
    )


def paths_to_dict(paths: Sequence[ExploratoryPath]) -> Dict[str, Any]:
    return {
        "schema": "exploratory_scenario_paths_v1",
        "path_count": len(paths),
        "paths": [p.to_dict() for p in paths],
    }
