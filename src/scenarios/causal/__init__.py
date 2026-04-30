"""
Causal and counterfactual reasoning scenario package.

Provides:
    * :class:`~src.scenarios.causal.taxonomy.CausalTheta`         — parameter vector for causal scenarios
    * :class:`~src.scenarios.causal.taxonomy.CausalThetaSampler`  — random / grid θ sampling
    * :class:`~src.scenarios.causal.generator.CausalScenarioGenerator` — scenario instantiation
"""

from src.scenarios.causal.generator import CausalScenarioGenerator
from src.scenarios.causal.taxonomy import CausalTheta, CausalThetaSampler

__all__ = [
    "CausalTheta",
    "CausalThetaSampler",
    "CausalScenarioGenerator",
]
