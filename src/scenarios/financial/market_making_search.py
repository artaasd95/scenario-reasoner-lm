"""
Market-making reasoning template search (S6-09).

Searches over reasoning_strategy_pool templates; scores template choice and
graph stats on fixtures — no order-book or exchange integration.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

from src.scenarios.financial.market_making_theta import MarketMakingReasoningTheta
from src.search.cards import AlgorithmCard, NodeCard, SearchOperator
from src.search.graph_monitor import SearchGraphMonitor

_REPO_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_FIXTURES_PATH = _REPO_ROOT / "data" / "scenarios" / "market_making_reasoning_fixtures.json"


@dataclass
class MarketMakingFixture:
    fixture_id: str
    theta: Dict[str, Any]
    expected_template: str
    goal: str = ""

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "MarketMakingFixture":
        return cls(
            fixture_id=data["fixture_id"],
            theta=data.get("theta", {}),
            expected_template=data.get("expected_template", ""),
            goal=data.get("goal", ""),
        )


def load_market_making_fixtures(path: Path | str | None = None) -> List[MarketMakingFixture]:
    fixture_path = Path(path) if path else DEFAULT_FIXTURES_PATH
    if not fixture_path.is_file():
        raise FileNotFoundError(f"Market making fixtures not found: {fixture_path}")
    payload = json.loads(fixture_path.read_text(encoding="utf-8"))
    return [MarketMakingFixture.from_dict(row) for row in payload.get("fixtures", [])]


@dataclass
class TemplateSearchResult:
    fixture_id: str
    chosen_template: str
    template_scores: Dict[str, float]
    search_graph: Dict[str, Any]
    theta: Dict[str, Any]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "fixture_id": self.fixture_id,
            "chosen_template": self.chosen_template,
            "template_scores": self.template_scores,
            "search_graph": self.search_graph,
            "theta": self.theta,
        }


def _score_template(
    template: str,
    theta: MarketMakingReasoningTheta,
) -> float:
    """Heuristic template fitness from inventory/spread regime (mock only)."""
    base = {
        "inventory_skew": 0.7,
        "adverse_selection": 0.75,
        "flow_toxicity": 0.65,
    }.get(template, 0.5)
    if theta.inventory_pressure == "high" and template == "inventory_skew":
        base += 0.2
    if theta.spread_regime == "wide" and template == "adverse_selection":
        base += 0.15
    return min(base, 1.0)


def search_templates(
    theta: MarketMakingReasoningTheta,
    *,
    fixture_id: str = "",
) -> TemplateSearchResult:
    """
    Template-pool search driven by MarketMakingReasoningTheta.

    Uses SearchGraphMonitor to record expansions over template nodes.
    """
    monitor = SearchGraphMonitor()
    algo = AlgorithmCard.new("template_search", SearchOperator.EXPAND, {"budget": theta.search_budget})
    root = NodeCard.new("template_root", depth=0, theta_slice=theta.to_dict())
    monitor.record_visit(root, algo.algorithm_id)

    scores: Dict[str, float] = {}
    best_template = theta.reasoning_strategy_pool[0]
    best_score = -1.0
    for template in theta.reasoning_strategy_pool:
        node = NodeCard.new(template, depth=1, parent_id=root.node_id)
        monitor.record_expansion(root, algo.algorithm_id)
        monitor.record_visit(node, algo.algorithm_id)
        score = _score_template(template, theta)
        scores[template] = round(score, 4)
        if score > best_score:
            best_score = score
            best_template = template
        if score < 0.55:
            monitor.record_prune(node, "below_threshold", algo.algorithm_id)

    if len(theta.reasoning_strategy_pool) > theta.search_budget:
        monitor.record_prune(root, "budget_exceeded", algo.algorithm_id)

    return TemplateSearchResult(
        fixture_id=fixture_id,
        chosen_template=best_template,
        template_scores=scores,
        search_graph=monitor.snapshot().to_dict(),
        theta=theta.to_dict(),
    )


def run_fixture(fixture: MarketMakingFixture) -> TemplateSearchResult:
    theta = MarketMakingReasoningTheta.from_dict(fixture.theta)
    result = search_templates(theta, fixture_id=fixture.fixture_id)
    return result


def score_fixture_result(
    fixture: MarketMakingFixture,
    result: TemplateSearchResult,
) -> Dict[str, float]:
    """Score template choice and graph stats for measurement fixtures."""
    template_match = 1.0 if result.chosen_template == fixture.expected_template else 0.0
    graph = result.search_graph
    return {
        "template_choice_match": template_match,
        "chosen_template_score": result.template_scores.get(result.chosen_template, 0.0),
        "graph_total_visits": float(graph.get("total_visits", 0)),
        "graph_total_prunes": float(graph.get("total_prunes", 0)),
        "graph_max_depth": float(graph.get("max_depth", 0)),
    }
