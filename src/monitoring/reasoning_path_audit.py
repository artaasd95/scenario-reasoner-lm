"""
Reasoning path audit metrics for simulation trace spans (S5-05).

Compares actual span chains to expected reasoning steps; optional Langfuse
integration no-ops when keys are absent (``tenk_demo_run`` pattern).
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence

from src.tracing.langfuse_client import LangfuseTracer, get_tracer
from src.tracing.trace_context import PARENT_TRACE_NAME, TraceSpanName


# Default enterprise simulation / demo span order
DEFAULT_ENTERPRISE_EXPECTED_SPANS: List[str] = [
    TraceSpanName.LOADING.value,
    TraceSpanName.CHUNKING.value,
    TraceSpanName.EXTRACTION.value,
    TraceSpanName.HYPOTHESES.value,
    TraceSpanName.CRITIQUE.value,
    TraceSpanName.RANKING.value,
]

DEFAULT_CAUSAL_EXPECTED_SPANS: List[str] = ["prompt", "reasoning", "answer"]


@dataclass
class PathAuditResult:
    expected_spans: List[str]
    actual_spans: List[str]
    path_fidelity: float
    missing_spans: List[str] = field(default_factory=list)
    extra_spans: List[str] = field(default_factory=list)
    langfuse_enabled: bool = False
    parent_trace_name: str = PARENT_TRACE_NAME

    def to_dict(self) -> Dict[str, Any]:
        return {
            "expected_spans": self.expected_spans,
            "actual_spans": self.actual_spans,
            "path_fidelity": self.path_fidelity,
            "missing_spans": self.missing_spans,
            "extra_spans": self.extra_spans,
            "langfuse_enabled": self.langfuse_enabled,
            "parent_trace_name": self.parent_trace_name,
        }


def compute_path_fidelity(
    actual_spans: Sequence[str],
    expected_spans: Sequence[str],
) -> PathAuditResult:
    """
    Score path fidelity as |expected ∩ actual| / |expected| (order-aware prefix bonus).

    Order bonus: +0.1 capped at 1.0 when the first min(len) spans match in order.
    """
    expected = list(expected_spans)
    actual = list(actual_spans)
    if not expected:
        return PathAuditResult(
            expected_spans=[],
            actual_spans=actual,
            path_fidelity=1.0 if not actual else 0.0,
        )

    expected_set = set(expected)
    actual_set = set(actual)
    overlap = len(expected_set & actual_set) / len(expected_set)
    prefix_len = min(len(expected), len(actual))
    order_bonus = 0.0
    if prefix_len:
        if all(expected[i] == actual[i] for i in range(prefix_len)):
            order_bonus = 0.1
        elif expected[0] == actual[0]:
            order_bonus = 0.05

    fidelity = min(overlap + order_bonus, 1.0)
    missing = [s for s in expected if s not in actual_set]
    extra = [s for s in actual if s not in expected_set]

    tracer = get_tracer()
    return PathAuditResult(
        expected_spans=expected,
        actual_spans=actual,
        path_fidelity=round(fidelity, 4),
        missing_spans=missing,
        extra_spans=extra,
        langfuse_enabled=tracer.enabled,
        parent_trace_name=PARENT_TRACE_NAME,
    )


def audit_simulation_spans(
    simulation_result: Dict[str, Any],
    *,
    expected_spans: Optional[List[str]] = None,
) -> PathAuditResult:
    """Audit spans from :meth:`~src.scenarios.simulation_runner.ScenarioSimulationRunner.run_fixture` output."""
    spans = simulation_result.get("spans", [])
    actual = [s.get("name", "") for s in spans if s.get("name")]
    if expected_spans is None:
        if simulation_result.get("scenario_type") == "causal":
            expected_spans = DEFAULT_CAUSAL_EXPECTED_SPANS
        else:
            expected_spans = DEFAULT_ENTERPRISE_EXPECTED_SPANS
    return compute_path_fidelity(actual, expected_spans)


def audit_langfuse_spans(
    tracer: Optional[LangfuseTracer] = None,
    *,
    expected_spans: Optional[List[str]] = None,
) -> PathAuditResult:
    """
    Audit in-memory spans recorded by :class:`~src.tracing.langfuse_client.LangfuseTracer`.

    No-op backend still records spans locally when keys are absent.
    """
    t = tracer or get_tracer()
    actual = [s.name for s in t.spans]
    expected = expected_spans or DEFAULT_ENTERPRISE_EXPECTED_SPANS
    result = compute_path_fidelity(actual, expected)
    result.langfuse_enabled = t.enabled
    return result


def manual_trace_review_instructions() -> str:
    """
    Template workflow for manual trace review (full sprint / live runs).

    CI remains smoke/mock only; use Langfuse UI when keys are set.
    """
    host = os.getenv("LANGFUSE_HOST", "https://cloud.langfuse.com").rstrip("/")
    return (
        "Manual trace review path:\n"
        f"1. Set LANGFUSE_PUBLIC_KEY and LANGFUSE_SECRET_KEY (optional).\n"
        f"2. Run enterprise demo or simulation with ALLOW_LIVE_PROVIDER=1 when approved.\n"
        f"3. Open {host}/trace/<trace_id> for parent trace `{PARENT_TRACE_NAME}`.\n"
        "4. Compare span order to expected: loading → chunking → extraction → "
        "hypotheses → critique → ranking.\n"
        "5. Use path_fidelity from audit_simulation_spans() for automated regression.\n"
        "CI: pytest tests/integration/test_reasoning_path_audit_smoke.py only (no live pipeline)."
    )
