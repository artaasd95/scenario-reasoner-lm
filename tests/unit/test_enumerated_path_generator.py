"""
Unit tests for enumerated scenario path generator (S6-02 / S7-02).
"""

from __future__ import annotations

import json

from src.risk.enterprise_theta import EnterpriseRiskTheta
from src.scenarios.causal.taxonomy import CausalTheta
from src.scenarios.enumerated_path_generator import (
    DEFAULT_ENTERPRISE_STAGES,
    ScenarioPathKind,
    generate_enumerated_paths,
    generate_from_fixture,
    load_enumerated_fixtures,
    paths_to_dict,
)
from src.scenarios.theta_mapping import enterprise_theta_to_causal_slice


class TestEnumeratedFixtures:
    def test_load_bundled_fixtures(self):
        fixtures = load_enumerated_fixtures()
        assert len(fixtures) >= 3
        types = {f.scenario_type for f in fixtures}
        assert ScenarioPathKind.ENTERPRISE in types
        assert ScenarioPathKind.CAUSAL in types

    def test_fixture_round_trip(self):
        for fx in load_enumerated_fixtures():
            paths = generate_from_fixture(fx, seed=42)
            assert len(paths) == fx.expected_path_count
            payload = paths_to_dict(paths)
            assert payload["path_count"] == fx.expected_path_count
            assert payload["schema"] == "enumerated_scenario_paths_v1"


class TestEnumeratedPathGenerator:
    def test_enterprise_cardinality_and_ordering(self):
        theta = EnterpriseRiskTheta(num_scenarios=5)
        paths = generate_enumerated_paths(
            theta,
            scenario_goal="Five catastrophic scenarios",
            scenario_type=ScenarioPathKind.ENTERPRISE,
        )
        assert len(paths) == 5
        stages = [p.stage for p in paths]
        assert stages == list(DEFAULT_ENTERPRISE_STAGES)
        assert [p.stage_index for p in paths] == list(range(5))

    def test_enterprise_maps_to_causal_slice(self):
        theta = EnterpriseRiskTheta()
        paths = generate_enumerated_paths(theta, scenario_type=ScenarioPathKind.ENTERPRISE)
        expected = enterprise_theta_to_causal_slice(theta).to_dict()
        for p in paths:
            assert p.causal_slice == expected
            assert p.theta["filing_id"] == "acme_corp_10k"

    def test_causal_good_bad_worst_ladder(self):
        theta = CausalTheta(chain_length=3, intervention_type="direct")
        paths = generate_enumerated_paths(
            theta,
            stages=["good", "bad", "worst"],
            scenario_type=ScenarioPathKind.CAUSAL,
            seed=42,
        )
        assert len(paths) == 3
        assert [p.stage for p in paths] == ["good", "bad", "worst"]
        assert all(p.metadata.get("answer") for p in paths)

    def test_three_stage_enterprise_ladder(self):
        fx = next(f for f in load_enumerated_fixtures() if f.fixture_id == "enterprise_good_bad_worst")
        paths = generate_from_fixture(fx)
        assert len(paths) == 3
        assert paths[0].stage == "good"
        assert paths[-1].stage == "worst"

    def test_paths_serializable(self):
        paths = generate_from_fixture(load_enumerated_fixtures()[0])
        serialized = json.dumps(paths_to_dict(paths))
        loaded = json.loads(serialized)
        assert loaded["path_count"] == len(paths)
