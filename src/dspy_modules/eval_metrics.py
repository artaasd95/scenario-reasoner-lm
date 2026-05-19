"""
Enterprise risk eval metrics (S2 criteria).

Loads ``data/eval/enterprise_risk_tiny.jsonl``, scores scenario cards against
golden labels and critique payloads, and aggregates per-criterion means.
No live LLM calls when used with offline demo output.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence

from src.dspy_modules.signatures import EVAL_CRITERIA
from src.dspy_modules.verify_scenarios import VerifyScenariosModule
from src.risk.schema import EnterpriseRiskScenarioCard

_REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_EVAL_PATH = _REPO_ROOT / "data" / "eval" / "enterprise_risk_tiny.jsonl"


@dataclass
class EvalRecord:
    """One row from the tiny enterprise eval set."""

    filing_id: str
    scenario_index: int
    expected_title_keywords: List[str] = field(default_factory=list)
    expected_severity: str = ""
    expected_likelihood: str = ""
    eval_criteria: Dict[str, float] = field(default_factory=dict)
    reviewer_notes: str = ""
    golden_title_substring: str = ""
    failure_mode_tags: List[str] = field(default_factory=list)
    rubric_hints: Dict[str, str] = field(default_factory=dict)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "EvalRecord":
        return cls(
            filing_id=data["filing_id"],
            scenario_index=int(data["scenario_index"]),
            expected_title_keywords=list(data.get("expected_title_keywords", [])),
            expected_severity=data.get("expected_severity", ""),
            expected_likelihood=data.get("expected_likelihood", ""),
            eval_criteria=dict(data.get("eval_criteria", {})),
            reviewer_notes=data.get("reviewer_notes", ""),
            golden_title_substring=data.get("golden_title_substring", ""),
            failure_mode_tags=list(data.get("failure_mode_tags", [])),
            rubric_hints=dict(data.get("rubric_hints", {})),
        )


def load_enterprise_eval_set(path: Path | str | None = None) -> List[EvalRecord]:
    """Load JSONL eval records; raises if path missing."""
    eval_path = Path(path) if path else DEFAULT_EVAL_PATH
    if not eval_path.is_file():
        raise FileNotFoundError(f"Eval set not found: {eval_path}")
    records: List[EvalRecord] = []
    with open(eval_path, encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            records.append(EvalRecord.from_dict(json.loads(line)))
    return records


def parse_rubric_thresholds(records: Sequence[EvalRecord]) -> Dict[str, float]:
    """Minimum threshold per criterion across eval rows (max of row mins)."""
    thresholds: Dict[str, float] = {}
    for record in records:
        for name, value in record.eval_criteria.items():
            thresholds[name] = max(thresholds.get(name, 0.0), float(value))
    return thresholds


def _title_matches(card: EnterpriseRiskScenarioCard, record: EvalRecord) -> bool:
    title_lower = card.title.lower()
    if record.golden_title_substring and record.golden_title_substring.lower() in title_lower:
        return True
    keywords = record.expected_title_keywords
    if not keywords:
        return True
    return all(kw.lower() in title_lower for kw in keywords)


def score_grounding(card: EnterpriseRiskScenarioCard, record: EvalRecord) -> float:
    has_evidence = bool(card.source_evidence)
    title_ok = _title_matches(card, record)
    if has_evidence and title_ok:
        return 0.95
    if has_evidence or title_ok:
        return 0.75
    return 0.35


def score_plausibility(card: EnterpriseRiskScenarioCard, critique: dict) -> float:
    if critique:
        return float(critique.get("plausibility_score", 0.0))
    chain_len = len(card.causal_chain)
    if chain_len >= 3 and card.missed_risk_rationale.strip():
        return 0.85
    return 0.5


def score_severity_clarity(card: EnterpriseRiskScenarioCard, record: EvalRecord) -> float:
    if record.expected_severity and card.severity == record.expected_severity:
        return 0.95
    if card.severity in ("catastrophic", "high"):
        return 0.8
    return 0.6


def score_non_duplication(
    card: EnterpriseRiskScenarioCard,
    all_cards: Sequence[EnterpriseRiskScenarioCard],
) -> float:
    titles = [c.title.lower().strip() for c in all_cards]
    if titles.count(card.title.lower().strip()) > 1:
        return 0.0
    return 1.0


def score_trace_completeness(card: EnterpriseRiskScenarioCard, run_trace_id: str = "") -> float:
    score = 0.0
    if card.trace_id:
        score += 0.5
    if run_trace_id:
        score += 0.5
    return min(score, 1.0)


def score_scenario(
    card: EnterpriseRiskScenarioCard,
    record: EvalRecord,
    critique: dict,
    all_cards: Sequence[EnterpriseRiskScenarioCard],
    run_trace_id: str = "",
) -> Dict[str, float]:
    """Per-criterion scores for one scenario card vs golden eval row."""
    return {
        "grounding": score_grounding(card, record),
        "plausibility": score_plausibility(card, critique),
        "severity_clarity": score_severity_clarity(card, record),
        "non_duplication": score_non_duplication(card, all_cards),
        "trace_completeness": score_trace_completeness(card, run_trace_id),
    }


def aggregate_criterion_scores(
  per_scenario: List[Dict[str, float]],
) -> Dict[str, float]:
    """Mean score per criterion across scenarios."""
    if not per_scenario:
        return {name: 0.0 for name in EVAL_CRITERIA}
    totals = {name: 0.0 for name in EVAL_CRITERIA}
    for scores in per_scenario:
        for name in EVAL_CRITERIA:
            totals[name] += float(scores.get(name, 0.0))
    n = len(per_scenario)
    return {name: round(totals[name] / n, 4) for name in EVAL_CRITERIA}


def evaluate_demo_result(
    demo_result: dict,
    eval_records: Optional[Sequence[EvalRecord]] = None,
) -> dict:
    """
    Score a ``run_enterprise_demo`` result dict against the tiny eval set.

    Matches scenarios by list index to ``scenario_index`` on eval rows.
    """
    records = list(eval_records or load_enterprise_eval_set())
    records_by_index = {r.scenario_index: r for r in records}
    scenario_dicts = demo_result.get("scenarios", [])
    critiques = demo_result.get("critiques", [])
    critique_by_trace = {c.get("trace_id"): c for c in critiques if c.get("trace_id")}

    cards = [EnterpriseRiskScenarioCard.from_dict(s) for s in scenario_dicts]
    run_trace_id = demo_result.get("trace_id", "")

    # Re-score critiques offline when missing
    verifier = VerifyScenariosModule(use_dspy=False)
    if not critiques and cards:
        critiques = verifier.critique(cards, {"evidence": []})
        critique_by_trace = {c.get("trace_id"): c for c in critiques}

    def _match_record(card: EnterpriseRiskScenarioCard) -> Optional[EvalRecord]:
        for record in records:
            if _title_matches(card, record):
                return record
        return records_by_index.get(
            next(
                (i for i, c in enumerate(cards) if c.trace_id == card.trace_id),
                -1,
            )
        )

    per_scenario: List[dict] = []
    per_scenario_scores: List[Dict[str, float]] = []

    for idx, card in enumerate(cards):
        record = _match_record(card) or records_by_index.get(idx)
        if record is None:
            continue
        critique = critique_by_trace.get(card.trace_id, critiques[idx] if idx < len(critiques) else {})
        scores = score_scenario(card, record, critique, cards, run_trace_id)
        per_scenario_scores.append(scores)
        per_scenario.append(
            {
                "scenario_index": record.scenario_index,
                "trace_id": card.trace_id,
                "title": card.title,
                "scores": scores,
                "thresholds": record.eval_criteria,
                "pass": all(
                    scores.get(k, 0.0) >= record.eval_criteria.get(k, 0.0)
                    for k in record.eval_criteria
                ),
            }
        )

    aggregate = aggregate_criterion_scores(per_scenario_scores)
    thresholds = parse_rubric_thresholds(records)
    return {
        "filing_id": demo_result.get("filing_id", ""),
        "run_trace_id": run_trace_id,
        "per_scenario": per_scenario,
        "aggregate_scores": aggregate,
        "thresholds": thresholds,
        "all_pass": all(
            aggregate.get(k, 0.0) >= thresholds.get(k, 0.0) for k in thresholds
        ),
    }
