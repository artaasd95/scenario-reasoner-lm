"""Unit tests for game bounded fixture + SearchGraphMonitor wiring (S6-07)."""

from __future__ import annotations

from src.scenarios.simulation_runner import (
    ScenarioSimulationRunner,
    load_game_bounded_default_fixture,
    load_simulation_fixtures,
)


class TestGameBoundedSimulation:
    def test_load_game_fixture(self):
        fx = load_game_bounded_default_fixture()
        assert fx.fixture_id == "game_bounded_default"
        assert fx.is_game_bounded

    def test_game_in_simulation_fixtures_list(self):
        fixtures = load_simulation_fixtures()
        ids = {f.fixture_id for f in fixtures}
        assert "game_bounded_default" in ids

    def test_run_records_search_graph(self):
        runner = ScenarioSimulationRunner(dry_run=True)
        fx = load_game_bounded_default_fixture()
        result = runner.run_fixture(fx)
        assert result["profile"] == "game_bounded"
        assert result["path_mode"] == "bounded"
        graph = result["search_graph"]
        assert graph["total_visits"] >= 1
        assert "total_expansions" in graph
        assert "total_prunes" in graph
