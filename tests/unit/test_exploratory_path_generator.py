"""
Unit tests for exploratory scenario path generator (S7-01).
"""

from __future__ import annotations

from src.scenarios.exploratory_path_generator import (
    ExploratoryPathConfig,
    ExploratoryStrategy,
    generate_exploratory_paths,
    generate_from_exploratory_fixture,
    load_exploratory_fixtures,
)


class TestExploratoryFixtures:
    def test_load_bundled_fixtures(self):
        fixtures = load_exploratory_fixtures()
        assert len(fixtures) >= 3
        strategies = {f.config.strategy for f in fixtures}
        assert ExploratoryStrategy.GRID in strategies
        assert ExploratoryStrategy.MONTE_CARLO in strategies

    def test_fixture_high_path_count(self):
        for fx in load_exploratory_fixtures():
            paths = generate_from_exploratory_fixture(fx)
            assert len(paths) >= fx.min_path_count


class TestExploratoryPathGenerator:
    def test_grid_deterministic(self):
        anchor = {"filing_id": "acme_corp_10k", "num_scenarios": 5, "severity_floor": "high"}
        grid = [
            {"filing_id": "acme_corp_10k", "num_scenarios": 3, "severity_floor": "high"},
            {"filing_id": "acme_corp_10k", "num_scenarios": 5, "severity_floor": "catastrophic"},
        ]
        cfg = ExploratoryPathConfig(max_paths=6, seed=42, strategy=ExploratoryStrategy.GRID)
        a = generate_exploratory_paths(anchor, config=cfg, theta_grid=grid, scenario_type="enterprise")
        b = generate_exploratory_paths(anchor, config=cfg, theta_grid=grid, scenario_type="enterprise")
        assert len(a) == len(b)
        assert [p.theta for p in a] == [p.theta for p in b]

    def test_monte_carlo_respects_cap(self):
        anchor = {
            "chain_length": 3,
            "intervention_type": "direct",
            "domain": "physical",
            "difficulty": "easy",
        }
        cfg = ExploratoryPathConfig(max_paths=5, seed=99, strategy=ExploratoryStrategy.MONTE_CARLO)
        paths = generate_exploratory_paths(anchor, config=cfg, scenario_type="causal")
        assert len(paths) == 5
        assert all(p.scenario_type == "causal" for p in paths)

    def test_tree_expand_causal(self):
        anchor = {
            "chain_length": 3,
            "intervention_type": "direct",
            "domain": "physical",
            "difficulty": "easy",
        }
        cfg = ExploratoryPathConfig(max_paths=10, seed=7, strategy=ExploratoryStrategy.TREE_EXPAND)
        paths = generate_exploratory_paths(anchor, config=cfg, scenario_type="causal")
        assert len(paths) >= 3
        assert len(paths) <= 10

    def test_monte_carlo_fixture_many_paths(self):
        fx = next(f for f in load_exploratory_fixtures() if f.fixture_id == "causal_monte_carlo")
        paths = generate_from_exploratory_fixture(fx)
        assert len(paths) >= 5
        indices = [p.path_index for p in paths]
        assert indices == list(range(len(paths)))
