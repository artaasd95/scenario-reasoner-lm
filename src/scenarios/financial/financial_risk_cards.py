"""
Financial risk scenario cards (S6-08).

Maps FinancialRiskTheta fixtures to enterprise-compatible scenario cards
on bundled samples — analysis framing only, not financial advice.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

from src.risk.schema import EnterpriseRiskScenarioCard, EvidenceChunk
from src.scenarios.financial.financial_risk_theta import FinancialRiskTheta
from src.scenarios.theta_mapping import financial_theta_to_enterprise_slice


def financial_theta_to_card(
    theta: FinancialRiskTheta,
    *,
    title: str,
    evidence: List[EvidenceChunk],
    causal_chain: List[str],
    missed_risk_rationale: str,
    trace_id: str = "",
) -> EnterpriseRiskScenarioCard:
    """Build an enterprise-compatible card from financial θ and bundled evidence."""
    ent = financial_theta_to_enterprise_slice(theta)
    severity = ent.severity_floor if ent.severity_floor != "high" else "high"
    if theta.stress_regime == "severe":
        severity = "catastrophic"
    elif theta.stress_regime == "adverse":
        severity = "high"

    horizon_map = {
        "0-6mo": "0-6 months",
        "6-18mo": "6-18 months",
        "18-36mo": "18-36 months",
        "36+mo": "36+ months",
    }
    horizon = horizon_map.get(theta.valuation_horizon, "18-36 months")
    rationale = (
        f"{missed_risk_rationale} "
        f"[lens={theta.risk_lens}; regime={theta.stress_regime}; "
        "research scenario — not investment advice.]"
    ).strip()
    return EnterpriseRiskScenarioCard(
        title=title,
        source_evidence=evidence,
        causal_chain=causal_chain,
        missed_risk_rationale=rationale,
        severity=severity,
        likelihood="medium",
        horizon=horizon,
        confidence=0.75,
        warning_signals=[
            f"risk_lens:{theta.risk_lens}",
            f"stress_regime:{theta.stress_regime}",
        ],
        mitigations=[],
        trace_id=trace_id or f"fin-{theta.filing_id}-{theta.risk_lens}",
    )


def cards_from_fixture_row(row: Dict[str, Any]) -> List[EnterpriseRiskScenarioCard]:
    """Build cards from a financial_risk_fixtures.jsonl row."""
    theta = FinancialRiskTheta.from_dict(row.get("theta", {}))
    evidence = [
        EvidenceChunk.from_dict(e) for e in row.get("evidence", [])
    ]
    cards: List[EnterpriseRiskScenarioCard] = []
    for spec in row.get("scenarios", []):
        cards.append(
            financial_theta_to_card(
                theta,
                title=spec["title"],
                evidence=evidence,
                causal_chain=list(spec.get("causal_chain", [])),
                missed_risk_rationale=spec.get("missed_risk_rationale", ""),
                trace_id=spec.get("trace_id", ""),
            )
        )
    return cards


def eval_rubric_scores(cards: List[EnterpriseRiskScenarioCard]) -> Dict[str, float]:
    """
    Score financial lens and stress_regime coverage without advice claims.

    Returns rubric slice keys for scenario_measurement reporting.
    """
    if not cards:
        return {
            "financial_lens_coverage": 0.0,
            "stress_regime_coverage": 0.0,
            "no_advice_claim_penalty": 1.0,
        }

    lenses = set()
    regimes = set()
    advice_penalty = 0.0
    import re

    banned_re = re.compile(
        r"\b(buy|sell|recommend|invest)\b|should invest|guaranteed return",
        re.IGNORECASE,
    )

    for card in cards:
        for sig in card.warning_signals:
            if sig.startswith("risk_lens:"):
                lenses.add(sig.split(":", 1)[1])
            if sig.startswith("stress_regime:"):
                regimes.add(sig.split(":", 1)[1])
        text = " ".join(
            [card.title, card.missed_risk_rationale]
            + card.causal_chain
            + [c.quote_text for c in card.source_evidence]
        )
        if banned_re.search(text):
            advice_penalty += 1.0

    n = len(cards)
    return {
        "financial_lens_coverage": min(len(lenses) / max(len(lenses), 1), 1.0),
        "stress_regime_coverage": min(len(regimes) / max(len(regimes), 1), 1.0),
        "no_advice_claim_penalty": round(advice_penalty / n, 4),
        "card_count": float(n),
    }
