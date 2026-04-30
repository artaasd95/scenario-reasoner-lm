"""
Robustness evaluator — assesses model quality across the θ-grid.

Runs inference on a trained model over a structured sweep of scenario
parameter combinations and reports per-θ metric breakdowns.  This surfaces
whether the model generalises across chain lengths, domains, difficulties, and
intervention types, or over-fits to the training distribution.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


class RobustnessEvaluator:
    """
    Evaluates a trained model across all combinations in a θ-grid.

    For each θ combination:
        1. Generate ``n_eval`` scenario instances.
        2. Run model inference to obtain (trace, answer) per instance.
        3. Score with the full :class:`~src.metrics.base_metrics.MetricRegistry`.
        4. Aggregate per-θ statistics.

    Args:
        model: Loaded causal LM (PEFT-wrapped or plain HuggingFace model).
        tokenizer: Corresponding tokenizer.
        metric_registry: Pre-populated :class:`~src.metrics.base_metrics.MetricRegistry`
            containing :class:`~src.metrics.causal_metrics.CausalChainAccuracy`,
            :class:`~src.metrics.causal_metrics.CounterfactualValidityScore`, and
            :class:`~src.metrics.causal_metrics.TrajectoryConsistency`.
        theta_grid: List of :class:`~src.scenarios.causal.taxonomy.CausalTheta`
            objects defining the evaluation sweep.
        n_eval: Number of instances to generate and evaluate per θ combination.
        max_new_tokens: Maximum tokens generated per instance.

    Example::

        from src.metrics.base_metrics import MetricRegistry
        from src.metrics.causal_metrics import (
            CausalChainAccuracy,
            CounterfactualValidityScore,
            TrajectoryConsistency,
        )

        registry = MetricRegistry()
        registry.register(CausalChainAccuracy())
        registry.register(CounterfactualValidityScore())
        registry.register(TrajectoryConsistency())

        evaluator = RobustnessEvaluator(model, tokenizer, registry, theta_grid)
        report = evaluator.evaluate()
    """

    def __init__(
        self,
        model,
        tokenizer,
        metric_registry,
        theta_grid: List[Any],
        n_eval: int = 50,
        max_new_tokens: int = 512,
    ) -> None:
        self.model = model
        self.tokenizer = tokenizer
        self.metric_registry = metric_registry
        self.theta_grid = theta_grid
        self.n_eval = n_eval
        self.max_new_tokens = max_new_tokens

    def evaluate(self) -> Dict[str, Any]:
        """
        Run the full per-θ evaluation sweep.

        Returns:
            Dict with keys:
                ``"per_theta"``  — list of per-θ result dicts
                ``"aggregate"``  — mean metrics across all θ combinations
                ``"theta_count"``— number of θ combinations evaluated
                ``"n_eval"``     — instances per θ combination
        """
        from src.scenarios.causal.generator import CausalScenarioGenerator

        generator = CausalScenarioGenerator()
        per_theta_results = []

        for theta in self.theta_grid:
            logger.info(
                "Evaluating θ: domain=%s chain=%d difficulty=%s type=%s",
                theta.domain,
                theta.chain_length,
                theta.difficulty,
                theta.intervention_type,
            )
            instances = generator.generate_batch(
                n=self.n_eval,
                theta_sampler=lambda t=theta: t,
            )

            predictions, references, intervention_types = self._run_inference(instances)

            self.metric_registry.reset_all()
            self.metric_registry.update_all(predictions, references)

            # CounterfactualValidityScore needs intervention_types kwarg
            if "counterfactual_validity" in self.metric_registry:
                self.metric_registry.get("counterfactual_validity").update(
                    predictions, references, intervention_types=intervention_types
                )

            scores = self.metric_registry.compute_all()

            per_theta_results.append({
                "theta": theta.to_dict(),
                "n_evaluated": len(instances),
                "metrics": scores,
            })

        aggregate = self._aggregate(per_theta_results)

        return {
            "per_theta": per_theta_results,
            "aggregate": aggregate,
            "theta_count": len(self.theta_grid),
            "n_eval": self.n_eval,
        }

    def save_report(self, report: Dict[str, Any], output_path: str) -> None:
        """
        Save an evaluation report as a formatted JSON file.

        Args:
            report: Dict returned by :meth:`evaluate`.
            output_path: Destination file path (will be created if absent).
        """
        path = Path(output_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8") as fh:
            json.dump(report, fh, indent=2, ensure_ascii=False)
        logger.info("Evaluation report saved to: %s", path)

    # ── Internals ─────────────────────────────────────────────────────────────

    def _run_inference(self, instances: List[Any]):
        """
        Run the model on each instance and return aligned (predictions, references, types).
        """
        import torch

        predictions: List[str] = []
        references: List[str] = []
        intervention_types: List[str] = []

        for inst in instances:
            prompt = inst.prompt
            inputs = self.tokenizer(
                prompt,
                return_tensors="pt",
                truncation=True,
                max_length=self.tokenizer.model_max_length - self.max_new_tokens,
            )
            input_ids = inputs["input_ids"].to(self.model.device)
            attention_mask = inputs["attention_mask"].to(self.model.device)

            with torch.no_grad():
                output_ids = self.model.generate(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    max_new_tokens=self.max_new_tokens,
                    do_sample=False,
                    pad_token_id=self.tokenizer.pad_token_id,
                )

            prompt_len = input_ids.shape[1]
            pred = self.tokenizer.decode(
                output_ids[0][prompt_len:], skip_special_tokens=True
            )
            predictions.append(pred)
            references.append(inst.answer or "")
            intervention_types.append(
                inst.metadata.get("intervention_type", "direct")
            )

        return predictions, references, intervention_types

    @staticmethod
    def _aggregate(per_theta_results: List[Dict[str, Any]]) -> Dict[str, float]:
        """Compute mean metric values across all θ combinations."""
        if not per_theta_results:
            return {}

        all_metrics: Dict[str, List[float]] = {}
        for result in per_theta_results:
            for metric_name, value in result["metrics"].items():
                if isinstance(value, (int, float)):
                    all_metrics.setdefault(metric_name, []).append(float(value))

        return {
            name: round(sum(vals) / len(vals), 4)
            for name, vals in all_metrics.items()
        }
