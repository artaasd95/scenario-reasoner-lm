"""
Causal/counterfactual scenario generator.

Given a :class:`~src.scenarios.causal.taxonomy.CausalTheta`, produces a
:class:`~src.scenarios.base_scenario.ScenarioInstance` containing:
    * A natural-language prompt (causal or counterfactual question)
    * A step-by-step CoT reasoning trace
    * The correct final answer

Implements :class:`~src.scenarios.base_scenario.ScenarioBase` so the generator
integrates naturally with the outer θ-parameter sweep described in
``docs/scenario-search-formulation.md``.
"""

from __future__ import annotations

import random
from typing import Any, Dict, List, Optional, Tuple

from src.scenarios.base_scenario import ScenarioBase, ScenarioInstance
from src.scenarios.causal.taxonomy import CausalTheta, CausalThetaSampler
from src.scenarios.causal.templates import (
    CAUSAL_VERBS,
    CF_QUESTION_TEMPLATES,
    CHAIN_LINK_TEMPLATES,
    COT_CF_CONCLUSION_CONFOUNDED,
    COT_CF_CONCLUSION_DIRECT,
    COT_CF_STEP_TEMPLATE,
    COT_CONCLUSION_TEMPLATE,
    COT_STEP_TEMPLATE,
    CONFOUNDER_TEMPLATES,
    DIRECT_QUESTION_TEMPLATES,
    DOMAIN_ENTITIES,
    SYSTEM_PROMPT,
)


class CausalScenarioGenerator(ScenarioBase[CausalTheta]):
    """
    Generates causal/counterfactual reasoning scenarios from CausalTheta parameters.

    Operationalizes the 6-tuple S = (X, Θ, T, A, R, Ω) for the causal domain:
        * Θ = :class:`CausalTheta`
          (chain_length, intervention_type, domain, difficulty, …)
        * X = partial causal chain state — sequence of (cause, effect) tuples
        * T = valid causal link extensions (no circular causation, within entity pool)
        * R = evaluated externally via :class:`~src.training.causal_reward.CausalRewardFunction`
        * Ω = no cycles, ``chain_length`` respected, confounder count respected

    All scenario text is generated from deterministic templates, so no LLM is
    required at dataset-creation time.  Template-based generation ensures:
        * Full coverage of the θ-grid with zero API cost
        * Reproducibility (given a seed)
        * Correct ground-truth answers for supervised and reward-based training

    Args:
        seed: Optional integer seed for reproducibility.
        sampler: Optional pre-configured :class:`CausalThetaSampler`.  A default
                 sampler covering all domains / difficulties is used if omitted.

    Example::

        gen = CausalScenarioGenerator(seed=42)

        # Single instance
        theta    = CausalTheta(chain_length=4, domain="physical",
                               difficulty="medium", intervention_type="counterfactual")
        instance = gen.instantiate(theta)
        print(instance.prompt)
        print(instance.reasoning_trace)
        print(instance.answer)

        # Batch with random θ sampling
        instances = gen.generate_batch(n=200)

        # Batch over explicit θ grid
        from src.scenarios.causal.taxonomy import CausalThetaSampler
        sampler  = CausalThetaSampler()
        grid     = sampler.grid(chain_lengths=[3, 5],
                                domains=["physical", "social"],
                                difficulties=["easy", "medium"])
        batch    = [gen.instantiate(t) for t in grid]
    """

    def __init__(
        self,
        seed: Optional[int] = None,
        sampler: Optional[CausalThetaSampler] = None,
    ) -> None:
        self._rng = random.Random(seed)
        self._sampler = sampler or CausalThetaSampler(seed=seed)

    # ── ScenarioBase interface ────────────────────────────────────────────────

    def sample_theta(self) -> CausalTheta:
        """Draw a :class:`CausalTheta` from the configured sampler."""
        return self._sampler.sample()

    def instantiate(self, theta: CausalTheta) -> ScenarioInstance:
        """
        Produce a complete scenario instance from *theta*.

        Steps:
            1. Select domain entities and build a causal chain of length θ.chain_length.
            2. Optionally pick a confounder entity.
            3. Construct the natural-language prompt.
            4. Generate the CoT reasoning trace.
            5. Derive the correct answer.

        Args:
            theta: Parameter vector from Θ.

        Returns:
            A :class:`ScenarioInstance` with ``prompt``, ``reasoning_trace``,
            ``answer``, and ``metadata`` populated.
        """
        chain = self._build_chain(theta)
        confounder = (
            self._pick_confounder(theta, chain) if theta.num_confounders > 0 else None
        )
        prompt = self._build_prompt(theta, chain, confounder)
        trace = self._build_cot_trace(theta, chain, confounder)
        answer = self._determine_answer(theta, chain, confounder)

        initial_state = f"{chain[0][0]} → ... → {chain[-1][1]}"
        return ScenarioInstance.new(
            theta=theta,
            initial_state=initial_state,
            prompt=prompt,
            reasoning_trace=trace,
            answer=answer,
            metadata={
                "chain": [(c, e) for c, e, _ in chain],
                "confounder": confounder,
                "domain": theta.domain,
                "intervention_type": theta.intervention_type,
                "difficulty": theta.difficulty,
                "chain_length": theta.chain_length,
            },
        )

    def is_valid_transition(
        self,
        state: str,
        action: str,
        next_state: str,
        theta: CausalTheta,
    ) -> bool:
        """
        Check Ω: next_state must be a domain entity not equal to current state (no cycles).

        Args:
            state: Current causal entity / state token.
            action: Causal verb string chosen by the policy.
            next_state: Proposed next causal entity.
            theta: Active parameter vector.

        Returns:
            ``True`` if the transition is valid under the causal domain rules.
        """
        entity_pool = DOMAIN_ENTITIES[theta.domain]
        return next_state in entity_pool and next_state != state

    def evaluate_trajectory(
        self,
        trajectory: List[str],
        theta: CausalTheta,
    ) -> float:
        """
        Compute a lightweight structural quality score for a reasoning trajectory.

        Scoring rubric:
            * Completeness (50 %) — step count relative to ``theta.chain_length``
            * Conclusion presence (30 %) — trajectory ends with a conclusion marker
            * No-cycle ratio (20 %) — fraction of unique states in the trajectory

        Args:
            trajectory: Ordered list of reasoning state strings.
            theta: Active parameter vector.

        Returns:
            Scalar quality score in [0, 1].
        """
        if not trajectory:
            return 0.0

        conclusion_keywords = (
            "therefore", "thus", "hence",
            "in conclusion", "the final outcome", "the answer",
        )
        has_conclusion = any(
            kw in step.lower()
            for step in trajectory
            for kw in conclusion_keywords
        )
        unique_ratio = len(set(trajectory)) / len(trajectory)
        completeness = min(len(trajectory) / max(theta.chain_length, 1), 1.0)

        return round(
            0.5 * completeness + 0.3 * float(has_conclusion) + 0.2 * unique_ratio,
            4,
        )

    # ── Internal construction helpers ────────────────────────────────────────

    def _build_chain(
        self,
        theta: CausalTheta,
    ) -> List[Tuple[str, str, str]]:
        """
        Build a causal chain as a list of (cause, effect, verb) triples.

        Chain length equals ``theta.chain_length``; every entity is unique to
        prevent trivial cycles.
        """
        pool = list(DOMAIN_ENTITIES[theta.domain])
        self._rng.shuffle(pool)
        n = min(theta.chain_length + 1, len(pool))
        entities = pool[:n]

        verb_pool = CAUSAL_VERBS[theta.difficulty]
        chain = []
        for i in range(len(entities) - 1):
            verb = self._rng.choice(verb_pool)
            chain.append((entities[i], entities[i + 1], verb))
        return chain

    def _pick_confounder(
        self,
        theta: CausalTheta,
        chain: List[Tuple[str, str, str]],
    ) -> Optional[str]:
        """
        Pick one confounder entity not already present in the causal chain.

        Returns ``None`` if the entity pool is exhausted.
        """
        chain_entities = {entity for pair in chain for entity in pair[:2]}
        pool = [e for e in DOMAIN_ENTITIES[theta.domain] if e not in chain_entities]
        return self._rng.choice(pool) if pool else None

    def _build_prompt(
        self,
        theta: CausalTheta,
        chain: List[Tuple[str, str, str]],
        confounder: Optional[str],
    ) -> str:
        """Construct the full LM prompt from the causal chain and θ."""
        # Build the numbered chain description
        chain_desc_lines = []
        for i, (cause, effect, verb) in enumerate(chain, start=1):
            template = self._rng.choice(CHAIN_LINK_TEMPLATES)
            sentence = template.format(cause=cause, verb=verb, effect=effect)
            chain_desc_lines.append(f"{i}. {sentence}")

        # Append confounder sentence if applicable
        if confounder and theta.num_confounders > 0:
            conf_template = self._rng.choice(CONFOUNDER_TEMPLATES)
            conf_sentence = conf_template.format(
                confounder=confounder,
                cause=chain[0][0],
                effect=chain[-1][1],
            )
            chain_desc_lines.append(conf_sentence)

        chain_description = "\n".join(chain_desc_lines)

        # Choose the question style
        if theta.intervention_type in ("counterfactual", "confounded"):
            removed_cause = chain[0][0]
            final_effect = chain[-1][1]
            q_templates = CF_QUESTION_TEMPLATES[theta.intervention_type]
            question = self._rng.choice(q_templates).format(
                removed_cause=removed_cause,
                final_effect=final_effect,
                confounder=confounder or "an external common cause",
            )
            body = f"{chain_description}\n\nQuestion: {question}"
        else:
            q_template = self._rng.choice(DIRECT_QUESTION_TEMPLATES)
            body = q_template.format(chain_description=chain_description)

        return f"{SYSTEM_PROMPT}\n\n{body}"

    def _build_cot_trace(
        self,
        theta: CausalTheta,
        chain: List[Tuple[str, str, str]],
        confounder: Optional[str],
    ) -> str:
        """
        Build a step-by-step CoT reasoning trace for the causal chain.

        For counterfactual/confounded scenarios the trace walks through the chain,
        marks the break point introduced by the intervention, and states the
        appropriate conclusion.
        """
        lines = []

        if theta.intervention_type in ("counterfactual", "confounded"):
            removed_cause = chain[0][0]
            final_effect = chain[-1][1]

            for i, (cause, effect, verb) in enumerate(chain, start=1):
                if i == 1:
                    line = COT_CF_STEP_TEMPLATE.format(
                        step_num=i,
                        removed_cause=removed_cause,
                    )
                else:
                    line = COT_STEP_TEMPLATE.format(
                        step_num=i, cause=cause, verb=verb, effect=effect
                    )
                lines.append(line)

            if theta.intervention_type == "confounded" and confounder:
                conclusion = COT_CF_CONCLUSION_CONFOUNDED.format(
                    removed_cause=removed_cause,
                    confounder=confounder,
                    final_effect=final_effect,
                )
            else:
                conclusion = COT_CF_CONCLUSION_DIRECT.format(
                    break_step=1,
                    final_effect=final_effect,
                )
            lines.append(conclusion)

        else:
            # Direct causal chain reasoning
            for i, (cause, effect, verb) in enumerate(chain, start=1):
                line = COT_STEP_TEMPLATE.format(
                    step_num=i, cause=cause, verb=verb, effect=effect
                )
                lines.append(line)
            lines.append(COT_CONCLUSION_TEMPLATE.format(final_effect=chain[-1][1]))

        return "\n".join(lines)

    def _determine_answer(
        self,
        theta: CausalTheta,
        chain: List[Tuple[str, str, str]],
        confounder: Optional[str],
    ) -> str:
        """Derive the correct ground-truth answer string for the scenario."""
        final_effect = chain[-1][1]

        if theta.intervention_type == "direct":
            return f"The final outcome is: {final_effect}."

        if theta.intervention_type == "counterfactual":
            return (
                f"No. If {chain[0][0]} had not occurred, the causal chain would be "
                f"broken and {final_effect} would NOT have happened."
            )

        # confounded
        if confounder:
            return (
                f"Yes. Even without {chain[0][0]}, the confounder '{confounder}' "
                f"would still independently drive {final_effect}, "
                f"so it would still occur."
            )
        return f"No. Without {chain[0][0]}, {final_effect} would not occur."
