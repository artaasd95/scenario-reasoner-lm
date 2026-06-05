"""Generator data source wrapping scenario fixtures (S9-DP-04)."""

from __future__ import annotations

from typing import Any, Dict, Iterator, List, Optional

from src.data.sources.base import DataSource
from src.scenarios.simulation_runner import ScenarioSimulationRunner, load_simulation_fixtures


class GeneratorDataSource(DataSource):
    def __init__(
        self,
        *,
        fixture_ids: Optional[List[str]] = None,
        include_game: bool = True,
    ) -> None:
        self.fixture_ids = fixture_ids
        self.include_game = include_game

    def iter_rows(self) -> Iterator[Dict[str, Any]]:
        runner = ScenarioSimulationRunner(dry_run=True)
        fixtures = load_simulation_fixtures()
        if self.fixture_ids:
            ids = set(self.fixture_ids)
            fixtures = [f for f in fixtures if f.fixture_id in ids]
        for fixture in fixtures:
            result = runner.run_fixture(fixture)
            yield {
                "path_id": fixture.fixture_id,
                "fixture_id": fixture.fixture_id,
                "scenario_type": result.get("scenario_type"),
                "path_mode": result.get("path_mode"),
                "theta": result.get("theta", {}),
                "theta_kind": "game" if fixture.is_game_bounded else fixture.scenario_type.value,
                "goal": fixture.goal,
                "search_graph": result.get("search_graph"),
                "primary_quality_score": 0.7,
            }
