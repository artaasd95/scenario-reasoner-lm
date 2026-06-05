"""
Evaluation entry point for Scenario Reasoner LM — Causal RLHF.

Usage::

    python scripts/evaluate.py \\
        --config experiments/configs/causal_rlhf_config.json \\
        --checkpoint experiments/results/causal_rlhf_run_01/dpo_checkpoint \\
        --output-dir experiments/results/causal_rlhf_run_01/eval \\
        [--n-eval 50]

Runs:
    1. Per-θ metric breakdown (CausalChainAccuracy, CounterfactualValidityScore,
       TrajectoryConsistency) across the configured θ-grid.
    2. Robustness sweep — reports per-θ and aggregate metric tables.
    3. Saves a JSON report to ``output_dir/robustness_report.json``.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

# Ensure project root is on sys.path when the script is run directly.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

logger = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Scenario Reasoner LM — Evaluation Script")
    parser.add_argument("--config", type=str, required=True, help="Path to JSON config file")
    parser.add_argument("--checkpoint", type=str, required=True,
                        help="Path to trained model checkpoint directory")
    parser.add_argument("--output-dir", type=str, default="./eval_outputs",
                        help="Output directory for reports and logs")
    parser.add_argument("--n-eval", type=int, default=50,
                        help="Number of instances to evaluate per θ combination")
    parser.add_argument(
        "--eval-config",
        "--data-config",
        dest="data_config",
        type=str,
        default=None,
        help="YAML eval data-platform config (e.g. configs/data/eval.yaml)",
    )
    return parser.parse_args()


def load_config(path: str) -> dict:
    with open(path) as f:
        return json.load(f)


def main() -> None:
    args = parse_args()
    config = load_config(args.config)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    from src.logging.local_logger import LocalLogger

    local_logger: LocalLogger | None = None
    try:
        local_logger = LocalLogger(
            name=config.get("experiment_name", "eval"),
            log_dir=str(output_dir / "logs"),
        )
        local_logger.log_config({**config, "checkpoint": args.checkpoint})

        if args.data_config:
            from src.data.pipeline.config import load_pipeline_config
            from src.data.pipeline.runner import PipelineRunner
            from src.data.pipeline.source_factory import build_source

            dp_config = load_pipeline_config(args.data_config)
            source = build_source(dp_config.source)
            pipeline_result = PipelineRunner(dp_config, source=source).run()
            local_logger.log_step(
                step=0,
                metrics={
                    "pipeline_measured": pipeline_result["stats"].get("measured", 0),
                    "datasource_id": dp_config.metadata.get("datasource_id"),
                },
                prefix="data",
            )

        logger.info("Loading model from checkpoint: %s", args.checkpoint)
        try:
            import torch
            from peft import PeftModel
            from transformers import AutoModelForCausalLM, AutoTokenizer
        except ImportError as exc:
            raise ImportError(
                "Evaluation requires 'transformers' and 'peft'. "
                "Install with: pip install transformers peft"
            ) from exc

        tokenizer = AutoTokenizer.from_pretrained(args.checkpoint, trust_remote_code=True)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        base_model = AutoModelForCausalLM.from_pretrained(
            config["model_name_or_path"],
            torch_dtype=torch.bfloat16,
            device_map="auto",
            trust_remote_code=True,
        )
        model = PeftModel.from_pretrained(base_model, args.checkpoint)
        model.eval()
        logger.info("Model loaded successfully")

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

        from src.scenarios.causal.taxonomy import CausalThetaSampler

        scenario_cfg = config.get("scenario", {})
        sampler = CausalThetaSampler()
        theta_grid = sampler.grid(
            chain_lengths=scenario_cfg.get("chain_lengths", [3, 5]),
            intervention_types=scenario_cfg.get("intervention_types"),
            domains=scenario_cfg.get("domains"),
            difficulties=scenario_cfg.get("difficulties"),
        )
        logger.info(
            "Evaluating over %d θ combinations × %d instances each",
            len(theta_grid),
            args.n_eval,
        )

        from src.evaluation.robustness_eval import RobustnessEvaluator

        evaluator = RobustnessEvaluator(
            model=model,
            tokenizer=tokenizer,
            metric_registry=registry,
            theta_grid=theta_grid,
            n_eval=args.n_eval,
        )
        report = evaluator.evaluate()

        report_path = str(output_dir / "robustness_report.json")
        evaluator.save_report(report, report_path)

        aggregate = report.get("aggregate", {})
        local_logger.log_step(step=0, metrics=aggregate, prefix="eval")

        logger.info("Aggregate metrics:")
        for metric_name, value in aggregate.items():
            logger.info("  %-35s %.4f", metric_name, value)

        logger.info("Evaluation complete. Report: %s", report_path)
    finally:
        if local_logger is not None:
            local_logger.close()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s  %(message)s")
    main()
