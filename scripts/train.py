"""
Training entry point for Scenario Reasoner LM — Causal RLHF (DPO).

Usage::

    python scripts/train.py \\
        --config experiments/configs/causal_rlhf_config.json \\
        --output-dir experiments/results/causal_rlhf_run_01 \\
        [--wandb]

    python scripts/train.py \\
        --config configs/training/unsloth_dpo_example.yaml \\
        [--wandb]

Pipeline:
    1. Generate causal scenarios (inline) or load distilled DPO pairs.
    2. Load model via training.backend (trl | unsloth).
    3. Build DPO preference pairs (inline) or use distilled JSONL.
    4. Run DPO training; optionally promote to SCENARIO_TRAINED_MODELS_ROOT.
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

logger = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Scenario Reasoner LM — Training Script")
    parser.add_argument("--config", type=str, required=True, help="JSON or YAML config file")
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Override output_dir from config",
    )
    parser.add_argument("--wandb", action="store_true", help="Enable Weights & Biases logging")
    parser.add_argument(
        "--data-config",
        type=str,
        default=None,
        help="YAML data-platform config (e.g. configs/data/train.yaml)",
    )
    parser.add_argument(
        "--llm-provider",
        type=str,
        default=None,
        help="Optional provider alias or config path for LLM-enhanced scenario generation.",
    )
    parser.add_argument(
        "--llm-model-id",
        type=str,
        default="demo-model",
        help="Model identifier used when --llm-provider is enabled.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    from src.training.config_loader import load_training_config
    from src.training.train_helpers import (
        build_inline_train_dataset,
        load_pipeline_train_jsonl,
        load_preference_data,
        resolve_training_config,
    )
    from src.training.backends import get_trainer_backend
    from src.training.artifacts import promote_checkpoint

    config = resolve_training_config(load_training_config(args.config))
    if args.llm_provider:
        config["runtime_llm_provider"] = args.llm_provider
        config["runtime_llm_model_id"] = args.llm_model_id

    output_dir = Path(args.output_dir or config.get("output_dir", "experiments/results"))
    output_dir.mkdir(parents=True, exist_ok=True)

    from src.logging.local_logger import LocalLogger
    from src.logging.wandb_logger import WandbLogger

    local_logger: LocalLogger | None = None
    wandb_logger: WandbLogger | None = None

    experiment_meta: dict = {
        "datasource_id": config.get("data", {}).get("datasource_id", "inline_generator"),
        "model_id": config.get("model_id"),
        "backend": config.get("training", {}).get("backend", "trl"),
    }

    try:
        local_logger = LocalLogger(
            name=config.get("experiment_name", "causal_rlhf"),
            log_dir=str(output_dir / "logs"),
        )
        local_logger.log_config(config)
        config["_local_logger"] = local_logger

        pipeline_jsonl_path: str | None = None
        if args.data_config:
            from src.data.pipeline.config import load_pipeline_config
            from src.data.pipeline.runner import PipelineRunner
            from src.data.pipeline.source_factory import build_source

            dp_config = load_pipeline_config(args.data_config)
            experiment_meta["datasource_id"] = dp_config.metadata.get("datasource_id", "pipeline")
            source = build_source(dp_config.source)
            pipeline_result = PipelineRunner(dp_config, source=source).run()
            pipeline_jsonl_path = pipeline_result.get("output")
            config.setdefault("training", {})["data_source"] = "pipeline"
            config["pipeline_train_jsonl"] = pipeline_jsonl_path
            local_logger.log_step(
                step=0,
                metrics={"pipeline_loaded": pipeline_result["stats"].get("loaded", 0)},
                prefix="data",
            )
            logger.info("Data pipeline artifact: %s", pipeline_jsonl_path)

        if args.wandb:
            wandb_logger = WandbLogger(
                project=config.get("project", "scenario-reasoner-lm"),
                name=config.get("experiment_name"),
                config=config,
            )

        training_cfg = config.get("training", {})
        data_source = training_cfg.get("data_source", "inline")
        train_dataset = None

        if data_source == "pipeline":
            jsonl_path = config.get("pipeline_train_jsonl") or pipeline_jsonl_path
            if not jsonl_path:
                raise ValueError("pipeline data_source requires --data-config or pipeline_train_jsonl")
            logger.info("Loading training data from pipeline JSONL: %s", jsonl_path)
            train_dataset = load_pipeline_train_jsonl(jsonl_path)
            n_theta = len({str(train_dataset[i].get("theta")) for i in range(len(train_dataset))})
            n_instances = len(train_dataset)
            logger.info("Loaded %d instances from pipeline (%d θ buckets)", n_instances, n_theta)
        elif data_source == "inline":
            logger.info("Generating causal scenarios from θ-grid ...")
            train_dataset, n_theta, n_instances = build_inline_train_dataset(config)
            logger.info(
                "Generated %d scenario instances across %d θ combinations",
                n_instances,
                n_theta,
            )

        backend_name = training_cfg.get("backend", "trl")
        backend = get_trainer_backend(backend_name)

        logger.info("Loading model (%s backend): %s", backend_name, config["model_name_or_path"])
        model, tokenizer = backend.load_model(config)

        if data_source == "distilled":
            logger.info("Loading distilled DPO pairs from manifest ...")
            preference_data = load_preference_data(config)
        else:
            logger.info("Building preference pairs (%d training scenarios) ...", len(train_dataset))
            preference_data = load_preference_data(
                config,
                model=model,
                tokenizer=tokenizer,
                train_dataset=train_dataset,
            )

        logger.info("Preference dataset size: %d pairs", len(preference_data))
        local_logger.log_step(
            step=0,
            metrics={
                "preference_pairs": len(preference_data),
                "datasource_id": experiment_meta.get("datasource_id"),
                "model_id": experiment_meta.get("model_id"),
                "backend": experiment_meta.get("backend"),
            },
            prefix="train",
        )

        checkpoint_path = backend.train(
            model,
            tokenizer,
            preference_data,
            config,
            use_wandb=args.wandb,
            output_dir=str(output_dir),
        )

        promoted = promote_checkpoint(checkpoint_path, config)
        if promoted:
            local_logger.log_step(
                step=1,
                metrics={"checkpoint": checkpoint_path, "promoted_to": promoted},
                prefix="train",
            )
        else:
            local_logger.log_step(
                step=1,
                metrics={"checkpoint": checkpoint_path},
                prefix="train",
            )

        logger.info("Training complete. Checkpoint: %s", checkpoint_path)
    finally:
        if local_logger is not None:
            local_logger.close()
        if wandb_logger is not None:
            wandb_logger.finish()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s  %(message)s")
    main()
