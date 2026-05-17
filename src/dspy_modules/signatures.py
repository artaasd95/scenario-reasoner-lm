"""
DSPy signatures for the enterprise risk 10-K pipeline.

Covers: evidence extraction, hypothesis generation, scenario building,
critique, and ranking.

Optimization: BootstrapFewShot first; MIPRO deferred until measurement exists.
"""

from __future__ import annotations

try:
    import dspy
except ImportError:  # pragma: no cover - optional dependency
    dspy = None  # type: ignore


def _require_dspy():
    if dspy is None:
        raise ImportError(
            "dspy is not installed. Install with: pip install dspy-ai"
        )


if dspy is not None:

    class ExtractEvidence(dspy.Signature):
        """Extract risk-relevant evidence quotes from 10-K chunks."""

        filing_excerpt: str = dspy.InputField(desc="Concatenated evidence chunk text")
        section_names: str = dspy.InputField(desc="Comma-separated SEC section names")
        evidence_json: str = dspy.OutputField(
            desc="JSON list of {chunk_id, section_name, quote, risk_theme}"
        )

    class GenerateHypotheses(dspy.Signature):
        """Propose missed or under-weighted enterprise risk hypotheses."""

        evidence_json: str = dspy.InputField()
        company_context: str = dspy.InputField()
        hypotheses_json: str = dspy.OutputField(
            desc="JSON list of {hypothesis, supporting_chunk_ids, rationale}"
        )

    class BuildScenario(dspy.Signature):
        """Build one catastrophic-but-plausible scenario card from a hypothesis."""

        hypothesis_json: str = dspy.InputField()
        evidence_json: str = dspy.InputField()
        scenario_json: str = dspy.OutputField(
            desc="JSON object matching EnterpriseRiskScenarioCard fields"
        )

    class CritiqueScenario(dspy.Signature):
        """Critique grounding, plausibility, and severity clarity."""

        scenario_json: str = dspy.InputField()
        evidence_json: str = dspy.InputField()
        critique_json: str = dspy.OutputField(
            desc="JSON {grounding_score, plausibility_score, severity_clarity, issues, suggestions}"
        )

    class RankScenarios(dspy.Signature):
        """Rank scenarios; penalize duplication."""

        scenarios_json: str = dspy.InputField(desc="JSON list of scenario objects")
        critiques_json: str = dspy.InputField()
        ranked_ids_json: str = dspy.OutputField(
            desc="JSON list of trace_id ordered best-first with composite scores"
        )

else:
    ExtractEvidence = None  # type: ignore
    GenerateHypotheses = None  # type: ignore
    BuildScenario = None  # type: ignore
    CritiqueScenario = None  # type: ignore
    RankScenarios = None  # type: ignore


# Evaluation criteria (used by verify_scenarios and tiny eval set)
EVAL_CRITERIA = (
    "grounding",
    "plausibility",
    "severity_clarity",
    "non_duplication",
    "trace_completeness",
)
