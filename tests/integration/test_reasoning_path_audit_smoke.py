"""
Smoke tests for reasoning path audit (S5-05).

No live pipeline or Langfuse keys required.
"""

from __future__ import annotations

from src.monitoring.reasoning_path_audit import (
    audit_simulation_spans,
    compute_path_fidelity,
    manual_trace_review_instructions,
)
from src.scenarios.simulation_runner import ScenarioSimulationRunner, load_simulation_fixtures
from src.tracing.langfuse_client import LangfuseTracer


class TestPathFidelity:
    def test_full_match_fidelity(self):
        expected = ["loading", "extraction", "ranking"]
        result = compute_path_fidelity(expected, expected)
        assert result.path_fidelity >= 0.9
        assert not result.missing_spans

    def test_partial_match_reports_missing(self):
        result = compute_path_fidelity(
            ["loading", "ranking"],
            ["loading", "extraction", "hypotheses", "ranking"],
        )
        assert "extraction" in result.missing_spans
        assert 0.0 < result.path_fidelity < 1.0

    def test_audit_simulation_spans(self):
        runner = ScenarioSimulationRunner(dry_run=True)
        fixture = load_simulation_fixtures()[0]
        sim = runner.run_fixture(fixture)
        audit = audit_simulation_spans(sim)
        assert audit.path_fidelity > 0.0
        assert len(audit.actual_spans) >= 1

    def test_langfuse_noop_tracer(self, monkeypatch):
        monkeypatch.delenv("LANGFUSE_PUBLIC_KEY", raising=False)
        monkeypatch.delenv("LANGFUSE_SECRET_KEY", raising=False)
        tracer = LangfuseTracer()
        assert tracer.enabled is False
        with tracer.span("loading", trace_id="test-trace") as rec:
            rec.outputs = {"ok": True}
        from src.monitoring.reasoning_path_audit import audit_langfuse_spans

        audit = audit_langfuse_spans(tracer, expected_spans=["loading"])
        assert audit.path_fidelity >= 0.9

    def test_manual_review_instructions(self):
        text = manual_trace_review_instructions()
        assert "tenk_demo_run" in text
        assert "pytest" in text
