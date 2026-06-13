"""LLM-enhanced scenario generation wrapper."""

from __future__ import annotations

import asyncio

from src.llm_integration.base import LLMProvider
from src.scenarios.base_scenario import ScenarioInstance
from src.scenarios.causal.generator import CausalScenarioGenerator


class LLMEnhancedScenarioGenerator:
    """Wraps deterministic generation and enriches reasoning text via provider."""

    def __init__(
        self,
        base_generator: CausalScenarioGenerator,
        llm_provider: LLMProvider,
        model_id: str,
    ) -> None:
        self._base = base_generator
        self._provider = llm_provider
        self._model_id = model_id

    def generate_batch(self, n: int, theta_sampler=None) -> list[ScenarioInstance]:
        instances = self._base.generate_batch(n=n, theta_sampler=theta_sampler)
        return [self._enhance_instance(inst) for inst in instances]

    def _enhance_instance(self, instance: ScenarioInstance) -> ScenarioInstance:
        prompt = (
            "Rewrite the reasoning trace with clearer causal explanation while preserving final answer.\n\n"
            f"Prompt:\n{instance.prompt}\n\n"
            f"Current reasoning:\n{instance.reasoning_trace or ''}\n\n"
            f"Answer:\n{instance.answer or ''}"
        )
        completion = asyncio.run(self._provider.complete(prompt, model_id=self._model_id))
        instance.reasoning_trace = completion.text
        instance.metadata["llm_provider"] = completion.backend_id
        instance.metadata["llm_model_id"] = self._model_id
        return instance
