"""
Abstract base class for scenario-type implementations.

Operationalizes the 6-tuple S = (X, Θ, T, A, R, Ω) from the project formulation
(see docs/scenario-search-formulation.md).

Each concrete scenario type (causal, code-debug, legal, …) inherits from
:class:`ScenarioBase` and implements the domain-specific logic for sampling
parameters, instantiating initial states, and evaluating trajectories.
"""

from __future__ import annotations

import uuid
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, Generic, List, Optional, TypeVar

ThetaType = TypeVar("ThetaType")


@dataclass
class ScenarioInstance:
    """
    A concrete scenario instance produced by instantiating θ.

    Represents a single node in the outer scenario-space:
        x_0 = g(θ), together with an optional full reasoning trajectory τ.

    Attributes:
        scenario_id: Unique identifier for this instance.
        theta: The parameter values (θ) that produced this instance.
        initial_state: The initial reasoning state x_0 = g(θ) (natural language).
        prompt: Full prompt to be fed to the language model.
        reasoning_trace: Optional ground-truth or sampled CoT/ToT trace.
        answer: Optional expected final answer at leaf node x_H.
        metadata: Arbitrary per-instance metadata (domain, difficulty, etc.).
    """

    scenario_id: str
    theta: Any
    initial_state: str
    prompt: str
    reasoning_trace: Optional[str] = None
    answer: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    @classmethod
    def new(
        cls,
        theta: Any,
        initial_state: str,
        prompt: str,
        **kwargs: Any,
    ) -> "ScenarioInstance":
        """
        Create a new instance with an auto-generated UUID.

        Args:
            theta: Parameter vector θ.
            initial_state: Natural-language description of x_0.
            prompt: LM prompt derived from x_0.
            **kwargs: Optional ``reasoning_trace``, ``answer``, ``metadata``.

        Returns:
            A fresh :class:`ScenarioInstance`.
        """
        return cls(
            scenario_id=str(uuid.uuid4()),
            theta=theta,
            initial_state=initial_state,
            prompt=prompt,
            **kwargs,
        )

    def to_dict(self) -> Dict[str, Any]:
        """Serialize the instance to a plain dict (JSON-serializable)."""
        theta_dict = (
            self.theta.__dict__ if hasattr(self.theta, "__dict__") else self.theta
        )
        return {
            "scenario_id": self.scenario_id,
            "theta": theta_dict,
            "initial_state": self.initial_state,
            "prompt": self.prompt,
            "reasoning_trace": self.reasoning_trace,
            "answer": self.answer,
            "metadata": self.metadata,
        }


class ScenarioBase(ABC, Generic[ThetaType]):
    """
    Abstract base class for scenario implementations.

    Each concrete subclass encodes one domain of reasoning by implementing the
    generative and evaluative components of the 6-tuple formulation:

        S = (X, Θ, T, A, R, Ω)

    where:
        X  — state space (intermediate reasoning/problem states)
        Θ  — parameter space (controls problem instantiation)
        T  — transition mechanism (valid state→state moves)
        A  — action/decision space used by the search policy
        R  — reward/utility functional
        Ω  — constraints (resource, safety, feasibility, domain rules)

    Subclasses must implement:
        * :meth:`sample_theta` — draw a parameter vector from Θ
        * :meth:`instantiate`  — compute x_0 = g(θ), build prompt + trace
        * :meth:`is_valid_transition` — check T and Ω for a state move
        * :meth:`evaluate_trajectory` — compute J(τ; θ) under R and Ω

    Example::

        class MyCausalScenario(ScenarioBase[MyCausalTheta]):
            def sample_theta(self): ...
            def instantiate(self, theta): ...
            def is_valid_transition(self, state, action, next_state, theta): ...
            def evaluate_trajectory(self, trajectory, theta): ...

        scenario = MyCausalScenario()
        instances = scenario.generate_batch(n=100)
    """

    @abstractmethod
    def sample_theta(self) -> ThetaType:
        """
        Draw a parameter vector θ from the parameter space Θ.

        Returns:
            A domain-specific parameter object (e.g. ``CausalTheta``).
        """

    @abstractmethod
    def instantiate(self, theta: ThetaType) -> ScenarioInstance:
        """
        Produce a concrete scenario instance from θ.

        Implements x_0 = g(θ): maps parameter values to an initial reasoning
        state and a language model prompt.

        Args:
            theta: Parameter vector from Θ.

        Returns:
            A fully populated :class:`ScenarioInstance`.
        """

    @abstractmethod
    def is_valid_transition(
        self,
        state: str,
        action: str,
        next_state: str,
        theta: ThetaType,
    ) -> bool:
        """
        Check whether a state transition is allowed under T and Ω.

        Args:
            state: Current reasoning state x_t.
            action: Action a_t chosen by the policy.
            next_state: Proposed next state x_{t+1}.
            theta: Active parameter vector.

        Returns:
            ``True`` if the transition is valid.
        """

    @abstractmethod
    def evaluate_trajectory(
        self,
        trajectory: List[str],
        theta: ThetaType,
    ) -> float:
        """
        Compute trajectory quality J(τ; θ).

        Args:
            trajectory: Ordered list of reasoning states [x_0, x_1, …, x_H].
            theta: Active parameter vector.

        Returns:
            Scalar quality score in [0, 1].
        """

    def generate_batch(
        self,
        n: int,
        theta_sampler: Optional[Callable[[], ThetaType]] = None,
    ) -> List[ScenarioInstance]:
        """
        Generate *n* scenario instances by sampling θ and calling :meth:`instantiate`.

        Args:
            n: Number of instances to generate.
            theta_sampler: Optional callable returning a ``ThetaType``.
                           Defaults to :meth:`sample_theta`.

        Returns:
            List of :class:`ScenarioInstance` objects.
        """
        sampler = theta_sampler if theta_sampler is not None else self.sample_theta
        return [self.instantiate(sampler()) for _ in range(n)]
