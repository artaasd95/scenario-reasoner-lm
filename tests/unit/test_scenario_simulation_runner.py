"""
Unit tests for bundled scenario simulation runner (S5-02).

Covers fixture load and dry-run path without network or paid keys.
"""

from __future__ import annotations

import pytest

from src.scenarios.resource_gate import allow_live_provider, assert_mock_or_gated
from src.scenarios.simulation_runner import (
    PathMode,
    ScenarioSimulationRunner,
    ScenarioType,
    load_simulation_fixtures,
)
from src.scenarios.theta_mapping import enterprise_theta_to_causal_slice
from src.risk.enterprise_theta import EnterpriseRiskTheta


class TestSimulationFixtures:
    def test_load_bundled_fixtures(self):
        fixtures = load_simulation_fixtures()
        assert len(fixtures) >= 4
        types = {f.scenario_type for f in fixtures}
        assert ScenarioType.ENTERPRISE in types
        assert ScenarioType.CAUSAL in types

    def test_wide_and_bounded_modes_present(self):
        fixtures = load_simulation_fixtures()
        modes = {f.path_mode for f in fixtures}
        assert PathMode.WIDE in modes
        assert PathMode.BOUNDED in modes


class TestScenarioSimulationRunner:
    def test_dry_run_all_fixtures(self):
        runner = ScenarioSimulationRunner(dry_run=True)
        report = runner.run_all()
        assert report["dry_run"] is True
        assert report["fixture_count"] == len(load_simulation_fixtures())
        assert report["provider_mode"] == "mock"

    def test_enterprise_bounded_result_shape(self):
        runner = ScenarioSimulationRunner(dry_run=True)
        fixtures = load_simulation_fixtures()
        ent = next(f for f in fixtures if f.fixture_id == "enterprise_bounded_default")
        result = runner.run_fixture(ent)
        assert result["scenario_type"] == "enterprise"
        assert result["path_mode"] == "bounded"
        assert result["scenario_count"] == 5
        assert len(result["spans"]) >= 1

    def test_causal_wide_many_paths(self):
        runner = ScenarioSimulationRunner(dry_run=True, seed=42)
        fixtures = load_simulation_fixtures()
        fx = next(f for f in fixtures if f.fixture_id == "causal_wide_monte_carlo")
        result = runner.run_fixture(fx)
        assert result["path_mode"] == "wide"
        assert result["path_count"] >= 3

    def test_filter_by_scenario_type(self):
        runner = ScenarioSimulationRunner(dry_run=True)
        report = runner.run_all(scenario_type="causal")
        for row in report["results"]:
            assert row["scenario_type"] == "causal"

    def test_live_blocked_without_gate(self, monkeypatch):
        monkeypatch.delenv("ALLOW_LIVE_PROVIDER", raising=False)
        runner = ScenarioSimulationRunner(dry_run=False, live=True)
        with pytest.raises(RuntimeError, match="ALLOW_LIVE_PROVIDER"):
            runner.run_all()


class TestResourceGate:
    def test_allow_live_default_false(self, monkeypatch):
        monkeypatch.delenv("ALLOW_LIVE_PROVIDER", raising=False)
        assert allow_live_provider() is False

    def test_assert_mock_or_gated_raises(self):
        with pytest.raises(RuntimeError):
            assert_mock_or_gated(live_requested=True)


class TestThetaMapping:
    def test_enterprise_to_causal_slice(self):
        theta = EnterpriseRiskTheta()
        causal = enterprise_theta_to_causal_slice(theta)
        assert 2 <= causal.chain_length <= 8
        assert causal.domain == "social"
