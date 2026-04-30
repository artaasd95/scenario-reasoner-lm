"""
Scenario implementations for Scenario Reasoner LM.

Each sub-package encodes one reasoning domain (causal, code-debug, legal, …)
by implementing the abstract :class:`~src.scenarios.base_scenario.ScenarioBase`
interface.  The 6-tuple formulation S = (X, Θ, T, A, R, Ω) from
``docs/scenario-search-formulation.md`` is operationalized here.
"""

from src.scenarios.base_scenario import ScenarioBase, ScenarioInstance

__all__ = ["ScenarioBase", "ScenarioInstance"]
