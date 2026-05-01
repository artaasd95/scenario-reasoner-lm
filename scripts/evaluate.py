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
    return parser.parse_args()


def load_config(path: str) -> dict:
    with open(path) as f:
        return json.load(f)


def main() -> None:
    args = parse_args()
    config = load_config(args.config)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # ── Logging setup ─────────────────────────────────────────────────────────
    from src.logging.local_logger import LocalLogger

    local_logger = LocalLogger(
        name=config.get("experiment_name", "eval"),
        log_dir=str(output_dir / "logs"),
    )
    local_logger.log_config({**config, "checkpoint": args.checkpoint})

    # ── Load model from checkpoint ────────────────────────────────────────────
    logger.info("Loading model from checkpoint: %s", args.checkpoint)
    try:
        import torch
        from peft import PeftModel
        from transformers import AutoModelForCausalLM, AutoTokenizer

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
    except ImportError as exc:
        raise ImportError(
            "Evaluation requires 'transformers' and 'peft'. "
            "Install with: pip install transformers peft"
        ) from exc

    # ── Build metric registry ─────────────────────────────────────────────────
    from src.metrics.base_metrics import MetricRegistry
    from src.metrics.causal_metrics import (
        CausalChainAccuracy,
        CounterfactualValidityScore,
        TrajectoryConsistency,
    )
    from src.monitoring.aha_monitor import AhaMonitor
    from src.monitoring.cot_monitor import CoTMonitor
    from src.monitoring.tot_monitor import ToTMonitor

    registry = MetricRegistry()
    registry.register(CausalChainAccuracy())
    registry.register(CounterfactualValidityScore())
    registry.register(TrajectoryConsistency())

    cot_monitor = CoTMonitor()
    tot_monitor = ToTMonitor()
    aha_monitor = AhaMonitor()

    # ── Build θ-grid ──────────────────────────────────────────────────────────
    from src.scenarios.causal.taxonomy import CausalThetaSampler

    scenario_cfg = config.get("scenario", {})
    sampler = CausalThetaSampler()
    theta_grid = sampler.grid(
        chain_lengths=scenario_cfg.get("chain_lengths", [3, 5]),
        intervention_types=scenario_cfg.get("intervention_types"),
        domains=scenario_cfg.get("domains"),
        difficulties=scenario_cfg.get("difficulties"),
    )
    logger.info("Evaluating over %d θ combinations × %d instances each",
                len(theta_grid), args.n_eval)

    # ── Run robustness evaluation ─────────────────────────────────────────────
    from src.evaluation.robustness_eval import RobustnessEvaluator

    evaluator = RobustnessEvaluator(
        model=model,
        tokenizer=tokenizer,
        metric_registry=registry,
        theta_grid=theta_grid,
        n_eval=args.n_eval,
    )
    report = evaluator.evaluate()

    # ── Save report ───────────────────────────────────────────────────────────
    report_path = str(output_dir / "robustness_report.json")
    evaluator.save_report(report, report_path)

    # ── Log aggregate metrics ─────────────────────────────────────────────────
    aggregate = report.get("aggregate", {})
    local_logger.log_metrics(aggregate, step=0)

    logger.info("Aggregate metrics:")
    for metric_name, value in aggregate.items():
        logger.info("  %-35s %.4f", metric_name, value)

    local_logger.close()
    logger.info("Evaluation complete. Report: %s", report_path)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s  %(message)s")
    main()
