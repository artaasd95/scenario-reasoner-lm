"""
Explicit mapping between enterprise and causal θ taxonomies (S5-01).

Preserves the headline benchmark (five grounded enterprise scenarios from one 10-K)
while documenting how ``EnterpriseRiskTheta`` parallels ``CausalTheta``.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

from src.risk.enterprise_theta import EnterpriseRiskTheta
from src.scenarios.causal.taxonomy import CausalTheta


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
