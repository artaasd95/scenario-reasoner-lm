"""ScenarioPathRow card tests (S9-DP-02)."""

from __future__ import annotations

from src.data.cards.scenario_path_row import ScenarioPathRow
from src.search.cards import NodeCard


class TestScenarioPathRow:
    def test_wraps_search_cards(self):
        node = NodeCard.new("state", depth=0)
        row = ScenarioPathRow(
            path_id="p1",
            nodes=[node],
            theta={"chain_length": 3, "intervention_type": "direct", "domain": "physical", "difficulty": "easy"},
            theta_kind="causal",
        )
        restored = ScenarioPathRow.from_dict(row.to_dict())
        assert restored.path_id == "p1"
        assert len(restored.nodes) == 1

    def test_normalized_theta(self):
        row = ScenarioPathRow(
            path_id="p2",
            theta={"chain_length": 2, "intervention_type": "direct", "domain": "physical", "difficulty": "easy"},
            theta_kind="causal",
        )
        theta = row.normalized_theta()
        assert theta.chain_length == 2
