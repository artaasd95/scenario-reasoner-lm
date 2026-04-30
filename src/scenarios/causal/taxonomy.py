"""
Parameter taxonomy (Θ) for causal/counterfactual scenario generation.

:class:`CausalTheta` defines the complete parameter space for causal scenarios.
:class:`CausalThetaSampler` provides random sampling and grid enumeration over Θ.

The parameter vector θ ∈ Θ controls:
    * The structural complexity of the causal graph (chain_length, entity_count)
    * The type of causal query (direct chain / confounded / counterfactual)
    * The subject-matter domain (physical, medical, social, mechanical)
    * The linguistic and logical difficulty (easy / medium / hard)
"""

from __future__ import annotations

import random
from dataclasses import dataclass
from itertools import product
from typing import Dict, List, Optional, Tuple


@dataclass
class CausalTheta:
    """
    Parameter vector θ for causal/counterfactual reasoning scenarios.

    Changing any field produces a distinct search problem in the formal sense:
    different initial state x_0 = g(θ), different action availability, and
    different terminal utility landscape.

    Attributes:
        chain_length: Number of causal links in the directed chain (2–8).
        intervention_type: Type of causal query.
            ``"direct"``        — straightforward A→B→C chain ending in a "what is the final outcome?" question.
            ``"confounded"``    — chain with one or more common-cause confounders, testing
                                  whether the model accounts for spurious correlation.
            ``"counterfactual"``— do-calculus-style intervention; asks
                                  "what if X had not occurred?"
        num_confounders: Number of confounding variables present (0–3).
            Only meaningful when ``intervention_type == "confounded"``.
        domain: Subject-matter domain of the scenario entities.
            One of ``"physical"``, ``"medical"``, ``"social"``, ``"mechanical"``.
        difficulty: Linguistic and structural complexity level.
            One of ``"easy"``, ``"medium"``, ``"hard"``.
        entity_count: Number of distinct named entities in the causal graph (2–5).
    """

    chain_length: int = 3
    intervention_type: str = "direct"
    num_confounders: int = 0
    domain: str = "physical"
    difficulty: str = "easy"
    entity_count: int = 2

    # Class-level constants (not included in dataclass comparisons by default)
    VALID_INTERVENTION_TYPES: tuple = ("direct", "confounded", "counterfactual")
    VALID_DOMAINS: tuple = ("physical", "medical", "social", "mechanical")
    VALID_DIFFICULTIES: tuple = ("easy", "medium", "hard")

    def __post_init__(self) -> None:
        if not (2 <= self.chain_length <= 8):
            raise ValueError(f"chain_length must be in [2, 8]; got {self.chain_length}")
        if self.intervention_type not in self.VALID_INTERVENTION_TYPES:
            raise ValueError(
                f"intervention_type must be one of {self.VALID_INTERVENTION_TYPES}; "
                f"got {self.intervention_type!r}"
            )
        if not (0 <= self.num_confounders <= 3):
            raise ValueError(
                f"num_confounders must be in [0, 3]; got {self.num_confounders}"
            )
        if self.domain not in self.VALID_DOMAINS:
            raise ValueError(
                f"domain must be one of {self.VALID_DOMAINS}; got {self.domain!r}"
            )
        if self.difficulty not in self.VALID_DIFFICULTIES:
            raise ValueError(
                f"difficulty must be one of {self.VALID_DIFFICULTIES}; "
                f"got {self.difficulty!r}"
            )
        if not (2 <= self.entity_count <= 5):
            raise ValueError(
                f"entity_count must be in [2, 5]; got {self.entity_count}"
            )

    def to_dict(self) -> Dict:
        return {
            "chain_length": self.chain_length,
            "intervention_type": self.intervention_type,
            "num_confounders": self.num_confounders,
            "domain": self.domain,
            "difficulty": self.difficulty,
            "entity_count": self.entity_count,
        }


class CausalThetaSampler:
    """
    Produces :class:`CausalTheta` instances either randomly or as an exhaustive grid.

    Args:
        chain_length_range: Inclusive ``(min, max)`` range for ``chain_length``.
        intervention_types: List of allowed intervention types.
            Defaults to all three.
        num_confounders_range: Inclusive ``(min, max)`` range for ``num_confounders``.
        domains: List of allowed domain strings.  Defaults to all four.
        difficulties: List of allowed difficulty strings.  Defaults to all three.
        entity_count_range: Inclusive ``(min, max)`` range for ``entity_count``.
        seed: Optional integer seed for reproducibility.

    Example::

        sampler = CausalThetaSampler(
            chain_length_range=(3, 6),
            domains=["physical", "social"],
            difficulties=["easy", "medium"],
            seed=42,
        )
        theta = sampler.sample()
        grid  = sampler.grid(chain_lengths=[3, 5], domains=["physical", "social"])
    """

    def __init__(
        self,
        chain_length_range: Tuple[int, int] = (2, 8),
        intervention_types: Optional[List[str]] = None,
        num_confounders_range: Tuple[int, int] = (0, 2),
        domains: Optional[List[str]] = None,
        difficulties: Optional[List[str]] = None,
        entity_count_range: Tuple[int, int] = (2, 4),
        seed: Optional[int] = None,
    ) -> None:
        self._rng = random.Random(seed)
        self.chain_length_range = chain_length_range
        self.intervention_types = intervention_types or list(
            CausalTheta.VALID_INTERVENTION_TYPES
        )
        self.num_confounders_range = num_confounders_range
        self.domains = domains or list(CausalTheta.VALID_DOMAINS)
        self.difficulties = difficulties or list(CausalTheta.VALID_DIFFICULTIES)
        self.entity_count_range = entity_count_range

    def sample(self) -> CausalTheta:
        """Draw one :class:`CausalTheta` uniformly at random within configured ranges."""
        intervention = self._rng.choice(self.intervention_types)
        confounders = (
            self._rng.randint(*self.num_confounders_range)
            if intervention == "confounded"
            else 0
        )
        return CausalTheta(
            chain_length=self._rng.randint(*self.chain_length_range),
            intervention_type=intervention,
            num_confounders=confounders,
            domain=self._rng.choice(self.domains),
            difficulty=self._rng.choice(self.difficulties),
            entity_count=self._rng.randint(*self.entity_count_range),
        )

    def grid(
        self,
        chain_lengths: Optional[List[int]] = None,
        intervention_types: Optional[List[str]] = None,
        domains: Optional[List[str]] = None,
        difficulties: Optional[List[str]] = None,
    ) -> List[CausalTheta]:
        """
        Return the full cross-product of provided axis values.

        Unspecified axes default to the first value in the sampler's configured list.

        Args:
            chain_lengths: List of ``chain_length`` values.
            intervention_types: List of ``intervention_type`` values.
            domains: List of ``domain`` values.
            difficulties: List of ``difficulty`` values.

        Returns:
            List of :class:`CausalTheta` covering every combination.
        """
        cl = chain_lengths or [self.chain_length_range[0]]
        it = intervention_types or [self.intervention_types[0]]
        dm = domains or self.domains
        df = difficulties or self.difficulties

        instances = []
        for chain_len, inv_type, domain, diff in product(cl, it, dm, df):
            confounders = 1 if inv_type == "confounded" else 0
            instances.append(
                CausalTheta(
                    chain_length=chain_len,
                    intervention_type=inv_type,
                    num_confounders=confounders,
                    domain=domain,
                    difficulty=diff,
                    entity_count=min(chain_len, 4),
                )
            )
        return instances
