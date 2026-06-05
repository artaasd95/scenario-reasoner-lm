"""Market-making template search tests (S6-09)."""

from __future__ import annotations

from src.scenarios.financial.market_making_search import (
    load_market_making_fixtures,
    run_fixture,
    score_fixture_result,
)


class TestMarketMakingReasoning:
    def test_load_fixtures(self):
        fixtures = load_market_making_fixtures()
        assert len(fixtures) >= 2

    def test_template_search_smoke(self):
        fixtures = load_market_making_fixtures()
        for fx in fixtures:
            result = run_fixture(fx)
            scores = score_fixture_result(fx, result)
            assert result.chosen_template in fx.theta.get("reasoning_strategy_pool", []) or result.chosen_template
            assert scores["graph_total_visits"] >= 1.0

    def test_inventory_fixture_prefers_skew(self):
        fixtures = load_market_making_fixtures()
        fx = next(f for f in fixtures if f.fixture_id == "mm_inventory_high")
        result = run_fixture(fx)
        assert result.chosen_template == "inventory_skew"
