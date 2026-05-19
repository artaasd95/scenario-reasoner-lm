"""
Scenario critique, ranking, and tiny evaluation set.

Evaluation criteria: grounding, plausibility, severity clarity,
non_duplication, trace_completeness.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from src.dspy_modules.signatures import EVAL_CRITERIA, CritiqueScenario, RankScenarios, _require_dspy
from src.risk.schema import EnterpriseRiskScenarioCard

_REPO_ROOT = Path(__file__).resolve().parents[2]
_EVAL_FIXTURE = _REPO_ROOT / "tests" / "fixtures" / "enterprise_risk_eval.json"
_EVAL_JSONL = _REPO_ROOT / "data" / "eval" / "enterprise_risk_tiny.jsonl"


_SEVERITY_ORDER = {"low": 0, "medium": 1, "high": 2, "catastrophic": 3}
_LIKELIHOOD_ORDER = {"low": 0, "medium": 1, "high": 2}


class VerifyScenariosModule:
    """Critique and rank scenario cards."""

    def __init__(self, use_dspy: bool = True) -> None:
        self.use_dspy = use_dspy
        self._critique = None
        self._rank = None
        if use_dspy:
            _require_dspy()
            import dspy

            self._critique = dspy.Predict(CritiqueScenario)
            self._rank = dspy.Predict(RankScenarios)

    def critique(
        self,
        cards: List[EnterpriseRiskScenarioCard],
        evidence_payload: dict,
        trace_callback: Optional[callable] = None,
    ) -> List[dict]:
        evidence_json = json.dumps(evidence_payload.get("evidence", []))
        critiques: List[dict] = []

        for card in cards:
            if self._critique is not None:
                result = self._critique(
                    scenario_json=card.to_json(),
                    evidence_json=evidence_json,
                )
                critique = json.loads(result.critique_json)
            else:
                critique = self._offline_critique(card)
            critique["trace_id"] = card.trace_id
            critiques.append(critique)

        if trace_callback:
            trace_callback(stage="critique", inputs={"n": len(cards)}, outputs={"n": len(critiques)})
        return critiques

    def rank(
        self,
        cards: List[EnterpriseRiskScenarioCard],
        critiques: List[dict],
        strategy: str = "severity_then_likelihood",
        trace_callback: Optional[callable] = None,
    ) -> List[EnterpriseRiskScenarioCard]:
        if self._rank is not None:
            scenarios_json = json.dumps([c.to_dict() for c in cards])
            critiques_json = json.dumps(critiques)
            result = self._rank(scenarios_json=scenarios_json, critiques_json=critiques_json)
            ranked_ids = json.loads(result.ranked_ids_json)
            by_id = {c.trace_id: c for c in cards}
            ranked = [by_id[tid] for tid in ranked_ids if tid in by_id]
            if len(ranked) < len(cards):
                ranked.extend(c for c in cards if c.trace_id not in ranked_ids)
        else:
            ranked = self._offline_rank(cards, critiques, strategy)

        if trace_callback:
            trace_callback(
                stage="ranking",
                inputs={"strategy": strategy},
                outputs={"order": [c.trace_id for c in ranked]},
            )
        return ranked

    @staticmethod
    def _offline_critique(card: EnterpriseRiskScenarioCard) -> dict:
        has_evidence = bool(card.source_evidence)
        return {
            "grounding_score": 0.9 if has_evidence else 0.3,
            "plausibility_score": 0.85,
            "severity_clarity": 0.9 if card.severity == "catastrophic" else 0.75,
            "non_duplication": 1.0,
            "trace_completeness": 1.0 if card.trace_id else 0.5,
            "issues": [],
            "suggestions": [],
        }

    @staticmethod
    def _offline_rank(
        cards: List[EnterpriseRiskScenarioCard],
        critiques: List[dict],
        strategy: str,
    ) -> List[EnterpriseRiskScenarioCard]:
        critique_by_id = {c["trace_id"]: c for c in critiques}

        def sort_key(card: EnterpriseRiskScenarioCard) -> Tuple:
            crit = critique_by_id.get(card.trace_id, {})
            if strategy == "composite_score":
                composite = (
                    crit.get("grounding_score", 0.5) * 0.3
                    + crit.get("plausibility_score", 0.5) * 0.25
                    + crit.get("severity_clarity", 0.5) * 0.2
                    + card.confidence * 0.25
                )
                return (-composite,)
            return (
                -_SEVERITY_ORDER.get(card.severity, 0),
                -_LIKELIHOOD_ORDER.get(card.likelihood, 0),
                -card.confidence,
            )

        return sorted(cards, key=sort_key)

    @staticmethod
    def load_eval_set() -> List[Dict[str, Any]]:
        """Tiny evaluation set for BootstrapFewShot / MIPRO optimizers."""
        if _EVAL_JSONL.is_file():
            rows: List[Dict[str, Any]] = []
            with open(_EVAL_JSONL, encoding="utf-8") as fh:
                for line in fh:
                    line = line.strip()
                    if line:
                        rows.append(json.loads(line))
            return rows
        if _EVAL_FIXTURE.is_file():
            with open(_EVAL_FIXTURE, encoding="utf-8") as fh:
                return json.load(fh)
        return []

    @staticmethod
    def score_against_criteria(
        card: EnterpriseRiskScenarioCard,
        critique: dict,
    ) -> Dict[str, float]:
        """Map critique fields to EVAL_CRITERIA names."""
        return {
            "grounding": float(critique.get("grounding_score", 0.0)),
            "plausibility": float(critique.get("plausibility_score", 0.0)),
            "severity_clarity": float(critique.get("severity_clarity", 0.0)),
            "non_duplication": float(critique.get("non_duplication", 1.0)),
            "trace_completeness": float(critique.get("trace_completeness", 0.0)),
        }
