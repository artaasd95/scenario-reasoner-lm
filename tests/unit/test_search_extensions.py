"""Unit tests for S6 search extensions (cards, game θ, graph monitor, financial θ)."""

from __future__ import annotations

import json

import pytest

from src.scenarios.financial.financial_risk_theta import FinancialRiskTheta
from src.scenarios.financial.market_making_theta import MarketMakingReasoningTheta
from src.scenarios.theta_mapping import (
    financial_theta_to_enterprise_slice,
    game_theta_to_causal_slice,
    s6_mapping_documentation,
)
from src.search.cards import ActionCard, AlgorithmCard, NodeCard, SearchOperator
from src.search.game_theta import GameTheoreticTheta, action_index_from_vector
from src.search.graph_monitor import SearchGraphMonitor
from src.search.manifold import ActionManifold, ManifoldKind


class TestAlgorithmCards:
    def test_node_action_algorithm_round_trip(self) -> None:
        node = NodeCard.new("root state", depth=0)
        action = ActionCard.new(
            node_id=node.node_id,
            stage=0,
            vector_slice=[0.1, 0.9, 0.0],
            label="expand",
            discrete_index=1,
        )
        algo = AlgorithmCard.new("mcts", SearchOperator.EXPAND, {"c": 1.4})
        meta = algo.apply_to(node)

        assert meta["node_id"] == node.node_id
        assert meta["operator"] == "expand"

        node2 = NodeCard.from_dict(node.to_dict())
        action2 = ActionCard.from_dict(action.to_dict())
        algo2 = AlgorithmCard.from_dict(algo.to_dict())

        assert node2.state_summary == node.state_summary
        assert action2.label == action.label
        assert algo2.operator == SearchOperator.EXPAND


class TestGameTheta:
    def test_default_vector_length(self) -> None:
        theta = GameTheoreticTheta.default(action_dim=10, num_stages=2)
        assert len(theta.action_vector) == 20

    def test_stage_slice_and_action_index(self) -> None:
        theta = GameTheoreticTheta.default(action_dim=4, num_stages=1)
        idx = theta.action_index_at_stage(0)
        assert 0 <= idx < theta.menu_size

    def test_action_index_from_vector(self) -> None:
        assert action_index_from_vector([0.0, 1.0, 0.0], menu_size=3) == 1

    def test_invalid_vector_length_raises(self) -> None:
        with pytest.raises(ValueError):
            GameTheoreticTheta(action_dim=3, num_stages=2, action_vector=[0.0] * 5)

    def test_json_round_trip(self) -> None:
        theta = GameTheoreticTheta.default()
        restored = GameTheoreticTheta.from_dict(theta.to_dict())
        assert restored.action_dim == theta.action_dim
        assert restored.num_stages == theta.num_stages


class TestManifold:
    def test_simplex_projects_to_unit_sum(self) -> None:
        m = ActionManifold(kind=ManifoldKind.SIMPLEX, dim=3)
        p = m.project([1.0, 2.0, 3.0])
        assert abs(sum(p) - 1.0) < 1e-6

    def test_box_clamps(self) -> None:
        m = ActionManifold(kind=ManifoldKind.BOX, dim=2)
        assert m.project([-1.0, 2.0]) == [0.0, 1.0]


class TestSearchGraphMonitor:
    def test_visit_expansion_prune_snapshot(self) -> None:
        monitor = SearchGraphMonitor()
        root = NodeCard.new("root", depth=0)
        child = NodeCard.new("child", depth=1, parent_id=root.node_id)
        algo_id = "search-1"

        monitor.record_visit(root, algo_id)
        monitor.record_expansion(root, algo_id)
        monitor.record_visit(child, algo_id)
        monitor.record_prune(child, "low_value", algo_id)

        report = monitor.snapshot()
        assert report.total_visits >= 2
        assert report.total_prunes == 1
        assert report.max_depth == 1
        assert report.branching_factor_estimate >= 0.0

        payload = json.dumps(report.to_dict())
        assert "branching_factor_estimate" in payload


class TestFinancialTheta:
    def test_financial_to_enterprise_slice(self) -> None:
        fin = FinancialRiskTheta(risk_lens="credit", stress_regime="adverse")
        ent = financial_theta_to_enterprise_slice(fin)
        assert ent.filing_id == fin.filing_id
        assert ent.num_scenarios == fin.num_scenarios

    def test_market_making_has_game_theta(self) -> None:
        mm = MarketMakingReasoningTheta()
        assert mm.game.action_dim == 10
        restored = MarketMakingReasoningTheta.from_dict(mm.to_dict())
        assert restored.search_budget == mm.search_budget

    def test_game_to_causal_slice(self) -> None:
        game = GameTheoreticTheta.default(num_stages=3)
        causal = game_theta_to_causal_slice(game)
        assert causal.chain_length >= 2

    def test_s6_mapping_doc(self) -> None:
        doc = s6_mapping_documentation()
        assert "game_to_causal" in doc
        assert "headline_benchmark_unchanged" in doc
