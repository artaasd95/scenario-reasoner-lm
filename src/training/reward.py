"""
Theta normalization at the reward boundary (SP-BUG-03).

Dataset rows may carry θ as dicts; reward scoring expects typed or mapping θ.
"""

from __future__ import annotations

from dataclasses import is_dataclass
from typing import Any, Dict, Mapping, Optional, Tuple, Union

from src.scenarios.causal.taxonomy import CausalTheta
from src.scenarios.financial.financial_risk_theta import FinancialRiskTheta
from src.scenarios.financial.market_making_theta import MarketMakingReasoningTheta
from src.search.game_theta import GameTheoreticTheta

ThetaLike = Union[
    CausalTheta,
    GameTheoreticTheta,
    FinancialRiskTheta,
    MarketMakingReasoningTheta,
    Mapping[str, Any],
]

def infer_theta_kind(data: Mapping[str, Any]) -> str:
    """Infer θ family from dict keys (first match wins)."""
    if "reasoning_strategy_pool" in data or "spread_regime" in data:
        return "market_making"
    if "action_dim" in data or "action_vector" in data:
        if "risk_lens" not in data and "filing_id" not in data:
            return "game"
    if "risk_lens" in data or "stress_regime" in data:
        return "financial_risk"
    if "chain_length" in data or "intervention_type" in data:
        return "causal"
    if "filing_id" in data and "num_scenarios" in data:
        return "financial_risk"
    return "unknown"


def normalize_theta(
    theta: Optional[ThetaLike],
    *,
    kind: Optional[str] = None,
) -> Tuple[Optional[Any], Optional[str]]:
    """
    Convert θ dict → dataclass at the reward boundary.

    Returns:
        (normalized_theta, error_reason). On success error_reason is None.
    """
    if theta is None:
        return None, None
    if is_dataclass(theta) and not isinstance(theta, type):
        return theta, None
    if not isinstance(theta, Mapping):
        return theta, None

    data = dict(theta)
    theta_kind = kind or infer_theta_kind(data)
    if theta_kind == "unknown":
        return None, "unrecognized_theta_schema"
    try:
        if theta_kind == "game":
            return GameTheoreticTheta.from_dict(data), None
        if theta_kind == "financial_risk":
            return FinancialRiskTheta.from_dict(data), None
        if theta_kind == "market_making":
            return MarketMakingReasoningTheta.from_dict(data), None
        return CausalTheta(**{k: v for k, v in data.items() if k in CausalTheta.__dataclass_fields__}), None
    except (TypeError, ValueError, KeyError) as exc:
        return None, f"invalid_{theta_kind}_theta: {exc}"


def theta_as_mapping(theta: Optional[Any]) -> Dict[str, Any]:
    """Serialize θ for logging regardless of input type."""
    if theta is None:
        return {}
    if hasattr(theta, "to_dict"):
        return theta.to_dict()
    if isinstance(theta, Mapping):
        return dict(theta)
    return {"repr": repr(theta)}
