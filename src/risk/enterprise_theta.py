"""
Enterprise risk parameter space (Θ) for 10-K demo runs.

Extends the repository's theta pattern (see ``CausalTheta``) without replacing
causal scenario abstractions in ``src.scenarios``.
"""

from __future__ import annotations

import random
from dataclasses import dataclass
from itertools import product
from typing import Dict, List, Optional, Tuple


@dataclass
class EnterpriseRiskTheta:
    """
    Parameter vector θ for enterprise risk scenario generation from a 10-K.

    Attributes:
        filing_id: Identifier for the source filing (bundled or uploaded).
        num_scenarios: Target number of ranked scenarios (demo default: 5).
        severity_floor: Minimum severity to surface (``low`` … ``catastrophic``).
        focus_sections: SEC sections to prioritize for evidence extraction.
        critique_passes: Number of critique rounds before ranking.
        ranking_strategy: ``"severity_then_likelihood"`` or ``"composite_score"``.
    """

    filing_id: str = "acme_corp_10k"
    num_scenarios: int = 5
    severity_floor: str = "high"
    focus_sections: Tuple[str, ...] = (
        "Risk Factors",
        "MD&A",
        "Business",
        "Legal Proceedings",
        "Cybersecurity",
        "Regulatory",
        "Supply Chain",
    )
    critique_passes: int = 1
    ranking_strategy: str = "severity_then_likelihood"

    VALID_SEVERITY_FLOORS: tuple = ("low", "medium", "high", "catastrophic")
    VALID_RANKING: tuple = ("severity_then_likelihood", "composite_score")

    def __post_init__(self) -> None:
        if not self.filing_id.strip():
            raise ValueError("filing_id must be non-empty")
        if not (1 <= self.num_scenarios <= 10):
            raise ValueError(f"num_scenarios must be in [1, 10]; got {self.num_scenarios}")
        if self.severity_floor not in self.VALID_SEVERITY_FLOORS:
            raise ValueError(
                f"severity_floor must be one of {self.VALID_SEVERITY_FLOORS}; "
                f"got {self.severity_floor!r}"
            )
        if self.critique_passes < 0:
            raise ValueError("critique_passes must be >= 0")
        if self.ranking_strategy not in self.VALID_RANKING:
            raise ValueError(
                f"ranking_strategy must be one of {self.VALID_RANKING}; "
                f"got {self.ranking_strategy!r}"
            )
        if not self.focus_sections:
            raise ValueError("focus_sections must be non-empty")

    def to_dict(self) -> Dict:
        return {
            "filing_id": self.filing_id,
            "num_scenarios": self.num_scenarios,
            "severity_floor": self.severity_floor,
            "focus_sections": list(self.focus_sections),
            "critique_passes": self.critique_passes,
            "ranking_strategy": self.ranking_strategy,
        }


class EnterpriseRiskThetaSampler:
    """Random and grid sampling over :class:`EnterpriseRiskTheta`."""

    DEFAULT_FILINGS = ("acme_corp_10k",)

    def __init__(
        self,
        filing_ids: Optional[List[str]] = None,
        num_scenarios_range: Tuple[int, int] = (5, 5),
        severity_floors: Optional[List[str]] = None,
        seed: Optional[int] = None,
    ) -> None:
        self._rng = random.Random(seed)
        self.filing_ids = filing_ids or list(self.DEFAULT_FILINGS)
        self.num_scenarios_range = num_scenarios_range
        self.severity_floors = severity_floors or ["high", "catastrophic"]

    def sample(self) -> EnterpriseRiskTheta:
        return EnterpriseRiskTheta(
            filing_id=self._rng.choice(self.filing_ids),
            num_scenarios=self._rng.randint(*self.num_scenarios_range),
            severity_floor=self._rng.choice(self.severity_floors),
        )

    def grid(
        self,
        filing_ids: Optional[List[str]] = None,
        num_scenarios_list: Optional[List[int]] = None,
        severity_floors: Optional[List[str]] = None,
    ) -> List[EnterpriseRiskTheta]:
        fids = filing_ids or self.filing_ids
        nums = num_scenarios_list or [self.num_scenarios_range[0]]
        floors = severity_floors or self.severity_floors
        return [
            EnterpriseRiskTheta(
                filing_id=fid,
                num_scenarios=n,
                severity_floor=floor,
            )
            for fid, n, floor in product(fids, nums, floors)
        ]

    def demo_default(self) -> EnterpriseRiskTheta:
        """Canonical θ for the bundled enterprise-risk demo."""
        return EnterpriseRiskTheta()
