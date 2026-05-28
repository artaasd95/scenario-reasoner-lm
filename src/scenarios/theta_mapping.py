"""
Explicit mapping between θ taxonomies (S5-01, S6 extensions).

Preserves the headline benchmark (five grounded enterprise scenarios from one 10-K)
while documenting how ``EnterpriseRiskTheta`` parallels ``CausalTheta``, and how
game-theoretic / financial θ relate without replacing source-of-truth types.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

from src.risk.enterprise_theta import EnterpriseRiskTheta
from src.scenarios.causal.taxonomy import CausalTheta
from src.scenarios.financial.financial_risk_theta import FinancialRiskTheta
from src.scenarios.financial.market_making_theta import MarketMakingReasoningTheta
from src.search.game_theta import GameTheoreticTheta


# Structural parallels (not 1:1 value equality — enterprise θ is filing-centric).
ENTERPRISE_TO_CAUSAL_AXIS_MAP: Dict[str, str] = {
    "filing_id": "domain",  # subject-matter anchor (filing vs causal domain)
    "num_scenarios": "chain_length",  # structural depth / cardinality
    "severity_floor": "difficulty",  # outcome strictness vs linguistic hardness
    "focus_sections": "entity_count",  # evidence breadth vs graph entities
    "critique_passes": "num_confounders",  # refinement rounds vs confounder depth
    "ranking_strategy": "intervention_type",  # selection policy vs query type
}


def enterprise_theta_to_causal_slice(theta: EnterpriseRiskTheta) -> CausalTheta:
    """
    Map enterprise θ to a causal θ *slice* for robustness reporting and search.

    The mapping is intentionally conservative: enterprise demos keep
    ``EnterpriseRiskTheta`` as source of truth; causal θ is derived for
    cross-domain measurement harnesses only.
    """
    severity_to_difficulty = {
        "low": "easy",
        "medium": "medium",
        "high": "hard",
        "catastrophic": "hard",
    }
    ranking_to_intervention = {
        "severity_then_likelihood": "direct",
        "composite_score": "confounded",
    }
    n_sections = len(theta.focus_sections)
    entity_count = min(max(n_sections // 2, 2), 5)
    chain_length = min(max(theta.num_scenarios, 2), 8)

    return CausalTheta(
        chain_length=chain_length,
        intervention_type=ranking_to_intervention.get(
            theta.ranking_strategy, "direct"
        ),
        num_confounders=min(theta.critique_passes, 3),
        domain="social",
        difficulty=severity_to_difficulty.get(theta.severity_floor, "medium"),
        entity_count=entity_count,
    )


def mapping_documentation() -> Dict[str, Any]:
    """Machine-readable mapping table for docs and measurement reports."""
    return {
        "headline_benchmark": (
            "Five source-grounded catastrophic enterprise scenarios from one bundled 10-K."
        ),
        "enterprise_source_of_truth": "EnterpriseRiskTheta",
        "causal_parallel": "CausalTheta",
        "axis_map": ENTERPRISE_TO_CAUSAL_AXIS_MAP,
        "notes": [
            "Enterprise θ controls filing ingestion and card ranking; causal θ controls template graphs.",
            "Do not replace causal RLHF paths with enterprise θ.",
            "Use causal slice only for θ-stratified robustness and cross-harness reporting.",
        ],
    }


def describe_mapping(theta: Optional[EnterpriseRiskTheta] = None) -> Dict[str, Any]:
    doc = mapping_documentation()
    if theta is not None:
        doc["example"] = {
            "enterprise": theta.to_dict(),
            "causal_slice": enterprise_theta_to_causal_slice(theta).to_dict(),
        }
    return doc


# S6: game-theoretic and financial parallels (reporting only).
GAME_TO_CAUSAL_AXIS_MAP: Dict[str, str] = {
    "action_dim": "entity_count",
    "num_stages": "chain_length",
    "menu_size": "num_confounders",
    "interaction_mode": "intervention_type",
    "manifold_kind": "domain",
}

FINANCIAL_TO_ENTERPRISE_AXIS_MAP: Dict[str, str] = {
    "risk_lens": "ranking_strategy",
    "stress_regime": "severity_floor",
    "valuation_horizon": "focus_sections",
}


def financial_theta_to_enterprise_slice(theta: FinancialRiskTheta) -> EnterpriseRiskTheta:
    """Financial risk θ reduces to enterprise headline θ."""
    return theta.to_enterprise_theta()


def game_theta_to_causal_slice(theta: GameTheoreticTheta) -> CausalTheta:
    """Conservative map for cross-harness robustness reporting."""
    interaction_to_type = {
        "single_agent": "direct",
        "two_player_zero_sum": "counterfactual",
        "multi_agent": "confounded",
    }
    chain_length = min(max(theta.num_stages, 2), 8)
    entity_count = min(max(theta.action_dim // 2, 2), 5)
    return CausalTheta(
        chain_length=chain_length,
        intervention_type=interaction_to_type.get(
            theta.interaction_mode.value, "direct"
        ),
        num_confounders=min(max(theta.menu_size, 1), 3),
        domain="social",
        difficulty="medium",
        entity_count=entity_count,
    )


def market_making_mapping_documentation() -> Dict[str, Any]:
    """How market-making reasoning θ composes game + financial extensions."""
    return {
        "purpose": "Search over reasoning templates; not live trading.",
        "game_theta_source_of_truth": "GameTheoreticTheta (staged action_vector)",
        "financial_parallel": "FinancialRiskTheta via filing-centric enterprise path",
        "strategy_pool": "reasoning_strategy_pool on MarketMakingReasoningTheta",
    }


def s6_mapping_documentation() -> Dict[str, Any]:
    """Machine-readable S6 extension map for docs and measurement reports."""
    return {
        "headline_benchmark_unchanged": mapping_documentation()["headline_benchmark"],
        "game_to_causal": GAME_TO_CAUSAL_AXIS_MAP,
        "financial_to_enterprise": FINANCIAL_TO_ENTERPRISE_AXIS_MAP,
        "market_making": market_making_mapping_documentation(),
        "notes": [
            "GameTheoreticTheta controls staged action vectors and manifold projection.",
            "FinancialRiskTheta extends EnterpriseRiskTheta; use enterprise slice for S2 demo.",
            "MarketMakingReasoningTheta searches reasoning templates under search_budget.",
        ],
    }
