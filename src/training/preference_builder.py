"""
DPO preference pair builder.

For each prompt, generates ``num_samples`` candidate traces from the model
at temperature > 0, scores all candidates via :class:`RewardComposer`, then
returns the highest-scoring trace as ``"chosen"`` and the lowest-scoring as
``"rejected"`` in DPO format.

This converts the monitoring-based reward signal into the pairwise preference
supervision that Direct Preference Optimization requires — with no human
annotation needed.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


class PreferenceBuilder:
    """
    Constructs ``(prompt, chosen, rejected)`` preference pairs for DPO training.

    Scoring is performed by :class:`~src.training.reward_composer.RewardComposer`.
    The model generates ``num_samples`` candidate continuations per prompt;
    the best and worst by ``R_total`` form the preference pair.

    Args:
        model: PEFT causal LM (output of :class:`~src.models.model_wrapper.ModelWrapper`).
        tokenizer: Tokenizer paired with ``model``.
        reward_composer: Configured :class:`~src.training.reward_composer.RewardComposer`.
        num_samples: Number of candidate traces generated per prompt.
            Higher values produce more reliable preference pairs but are slower.
        temperature: Sampling temperature.  Values in ``(0.6, 1.0)`` work well.
        max_new_tokens: Maximum number of new tokens generated per candidate.

    Example::

        builder = PreferenceBuilder(model, tokenizer, reward_composer, num_samples=4)

        # Single pair
        pair = builder.build_pair(prompt, expected_answer=answer, theta=theta)

        # Full dataset
        pairs = builder.build_dataset(prompts, expected_answers, thetas)
    """

    def __init__(
        self,
        model,
        tokenizer,
        reward_composer,
        num_samples: int = 4,
        temperature: float = 0.8,
        max_new_tokens: int = 512,
    ) -> None:
        self.model = model
        self.tokenizer = tokenizer
        self.reward_composer = reward_composer
        self.num_samples = num_samples
        self.temperature = temperature
        self.max_new_tokens = max_new_tokens

    def build_pair(
        self,
        prompt: str,
        expected_answer: Optional[str] = None,
        theta: Optional[Any] = None,
    ) -> Dict[str, str]:
        """
        Generate one ``(prompt, chosen, rejected)`` preference pair.

        Args:
            prompt: The scenario prompt to complete.
            expected_answer: Optional ground-truth answer for reward scoring.
            theta: Optional :class:`~src.scenarios.causal.taxonomy.CausalTheta`
                   for domain-specific reward scoring.

        Returns:
            Dict with ``"prompt"``, ``"chosen"``, ``"rejected"`` string keys.

        Raises:
            RuntimeError: If fewer than 2 candidates are generated.
        """
        candidates = self._generate_candidates(prompt)

        if len(candidates) < 2:
            raise RuntimeError(
                f"Need at least 2 candidate traces to form a preference pair; "
                f"got {len(candidates)}."
            )

        scores = [
            self.reward_composer.score(
                prompt=prompt,
                trace=cand,
                answer=cand,
                expected_answer=expected_answer,
                theta=theta,
                sample_id=i,
            )["R_total"]
            for i, cand in enumerate(candidates)
        ]

        best_idx = max(range(len(scores)), key=lambda i: scores[i])
        worst_idx = min(range(len(scores)), key=lambda i: scores[i])

        logger.debug(
            "Preference pair scores — chosen: %.4f, rejected: %.4f",
            scores[best_idx],
            scores[worst_idx],
        )

        return {
            "prompt": prompt,
            "chosen": candidates[best_idx],
            "rejected": candidates[worst_idx],
        }

    def build_dataset(
        self,
        prompts: List[str],
        expected_answers: Optional[List[Optional[str]]] = None,
        thetas: Optional[List[Any]] = None,
    ) -> List[Dict[str, str]]:
        """
        Build a full preference dataset from a list of prompts.

        Prompts that raise an exception during generation are skipped with a
        warning rather than aborting the whole build.

        Args:
            prompts: List of scenario prompts.
            expected_answers: Optional per-prompt ground-truth answers.
            thetas: Optional per-prompt :class:`CausalTheta` objects.

        Returns:
            List of ``{"prompt", "chosen", "rejected"}`` dicts.
        """
        n = len(prompts)
        expected_answers = expected_answers or [None] * n
        thetas = thetas or [None] * n

        pairs = []
        for i, (prompt, ea, theta) in enumerate(zip(prompts, expected_answers, thetas)):
            try:
                pair = self.build_pair(prompt, expected_answer=ea, theta=theta)
                pairs.append(pair)
            except Exception as exc:
                logger.warning("Skipping prompt %d during preference build: %s", i, exc)

        logger.info(
            "Built %d preference pairs from %d prompts (%d skipped)",
            len(pairs),
            n,
            n - len(pairs),
        )
        return pairs

    # ── Internals ─────────────────────────────────────────────────────────────

    def _generate_candidates(self, prompt: str) -> List[str]:
        """
        Run the model ``num_samples`` times at ``temperature`` to produce diverse
        candidate continuations.

        Returns:
            List of decoded candidate strings (prompt stripped).
        """
        import torch

        inputs = self.tokenizer(
            prompt,
            return_tensors="pt",
            truncation=True,
            max_length=self.tokenizer.model_max_length - self.max_new_tokens,
        )
        input_ids = inputs["input_ids"].to(self.model.device)
        attention_mask = inputs["attention_mask"].to(self.model.device)

        with torch.no_grad():
            outputs = self.model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                max_new_tokens=self.max_new_tokens,
                do_sample=True,
                temperature=self.temperature,
                num_return_sequences=self.num_samples,
                pad_token_id=self.tokenizer.pad_token_id,
            )

        prompt_len = input_ids.shape[1]
        return [
            self.tokenizer.decode(out[prompt_len:], skip_special_tokens=True)
            for out in outputs
        ]
