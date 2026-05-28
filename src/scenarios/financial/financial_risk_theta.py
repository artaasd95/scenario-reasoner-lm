"""
Financial risk analysis θ (S6).

Extends enterprise filing-centric parameters with risk lenses and stress regimes.
Output cards remain compatible with ``EnterpriseRiskScenarioCard`` where possible.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List

from src.risk.enterprise_theta import EnterpriseRiskTheta


@dataclass
class FinancialRiskTheta:
    """
    Parameters for financial risk analysis scenarios.

    Wraps enterprise θ axes plus financial-specific controls.
    """

    filing_id: str = "acme_corp_10k"
    num_scenarios: int = 5
    severity_floor: str = "high"
    focus_sections: List[str] = field(
        default_factory=lambda: ["Risk Factors", "MD&A"]
    )
    critique_passes: int = 1
    ranking_strategy: str = "severity_then_likelihood"
    risk_lens: str = "market"  # credit | liquidity | operational | market
    stress_regime: str = "baseline"  # baseline | adverse | severe
    valuation_horizon: str = "18-36mo"
    seed: int = 42

    def to_enterprise_theta(self) -> EnterpriseRiskTheta:
        """Slice to enterprise demo θ (headline benchmark path)."""
        return EnterpriseRiskTheta(
            filing_id=self.filing_id,
            num_scenarios=self.num_scenarios,
            severity_floor=self.severity_floor,
            focus_sections=tuple(self.focus_sections),
            critique_passes=self.critique_passes,
            ranking_strategy=self.ranking_strategy,
        )

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "FinancialRiskTheta":
        return cls(**{k: v for k, v in data.items() if k in cls.__dataclass_fields__})
