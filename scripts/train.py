"""
Training entry point for Scenario Reasoner LM — Causal RLHF (DPO).

Usage::

    python scripts/train.py \\
        --config experiments/configs/causal_rlhf_config.json \\
        --output-dir experiments/results/causal_rlhf_run_01 \\
        [--wandb]

Pipeline:
    1. Generate causal scenarios from the θ-grid specified in config.
    2. Load / fine-tune a 7B QLoRA model via ModelWrapper.
    3. Build DPO preference pairs using PreferenceBuilder + RewardComposer.
    4. Run DPO training via RLHFTrainer (TRL backend).
    5. Save checkpoint and per-step metrics.
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

logger = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Scenario Reasoner LM — Training Script")
    parser.add_argument("--config", type=str, required=True, help="Path to JSON config file")
    parser.add_argument("--output-dir", type=str, default=None,
                        help="Override output_dir from config")
    parser.add_argument("--wandb", action="store_true", help="Enable Weights & Biases logging")
    return parser.parse_args()


def load_config(path: str) -> dict:
    with open(path) as f:
        return json.load(f)


def main() -> None:
    args = parse_args()
    config = load_config(args.config)

    output_dir = Path(args.output_dir or config.get("output_dir", "experiments/results"))
    output_dir.mkdir(parents=True, exist_ok=True)

    # ── Logging setup ─────────────────────────────────────────────────────────
    from src.logging.local_logger import LocalLogger
    from src.logging.wandb_logger import WandbLogger

    local_logger = LocalLogger(
        name=config.get("experiment_name", "causal_rlhf"),
        log_dir=str(output_dir / "logs"),
    )
    local_logger.log_config(config)

    wandb_logger: WandbLogger | None = None
    if args.wandb:
        wandb_logger = WandbLogger(
            project=config.get("project", "scenario-reasoner-lm"),
            name=config.get("experiment_name"),
            config=config,
        )

    # ── Step 1: Generate causal scenarios ────────────────────────────────────
    logger.info("Generating causal scenarios from θ-grid ...")
    from src.scenarios.causal.generator import CausalScenarioGenerator
    from src.scenarios.causal.taxonomy import CausalThetaSampler
    from src.data.causal_dataset import CausalReasoningDataset

    scenario_cfg = config.get("scenario", {})
    sampler = CausalThetaSampler(
        chain_length_range=(
            min(scenario_cfg.get("chain_lengths", [3])),
            max(scenario_cfg.get("chain_lengths", [5])),
        ),
        intervention_types=scenario_cfg.get("intervention_types"),
        domains=scenario_cfg.get("domains"),
        difficulties=scenario_cfg.get("difficulties"),
        seed=config.get("data", {}).get("seed", 42),
    )
    generator = CausalScenarioGenerator(
        seed=config.get("data", {}).get("seed", 42),
        sampler=sampler,
    )
    theta_grid = sampler.grid(
        chain_lengths=scenario_cfg.get("chain_lengths", [3, 5]),
        intervention_types=scenario_cfg.get("intervention_types"),
        domains=scenario_cfg.get("domains"),
        difficulties=scenario_cfg.get("difficulties"),
    )
    n_per_combo = scenario_cfg.get("n_per_combo", 200)
    all_instances = []
    for theta in theta_grid:
        all_instances.extend(generator.generate_batch(n=n_per_combo,
                                                       theta_sampler=lambda t=theta: t))

    logger.info("Generated %d scenario instances across %d θ combinations",
                len(all_instances), len(theta_grid))

    full_dataset = CausalReasoningDataset.from_scenario_instances(all_instances)
    train_dataset, _ = full_dataset.stratified_split(
        train_frac=config.get("data", {}).get("train_frac", 0.9)
    )

    # ── Step 2: Load model ────────────────────────────────────────────────────
    logger.info("Loading model: %s", config["model_name_or_path"])
    from src.models.model_wrapper import ModelWrapper

    wrapper = ModelWrapper.from_config(config)
    model, tokenizer = wrapper.load()

    # ── Step 3: Build DPO preference pairs ───────────────────────────────────
    logger.info("Building preference pairs (%d training scenarios) ...",
                len(train_dataset))
    from src.training.causal_reward import CausalRewardFunction
    from src.training.reward_composer import RewardComposer
    from src.training.preference_builder import PreferenceBuilder

    reward_weights = config.get("reward_weights", {})
    reward_fn = CausalRewardFunction()
    composer = RewardComposer(
        task_reward_fn=reward_fn,
        alpha=reward_weights.get("alpha_cot", 0.15),
        beta=reward_weights.get("beta_tot", 0.10),
        gamma=reward_weights.get("gamma_aha", 0.05),
    )

    pb_cfg = config.get("preference_builder", {})
    builder = PreferenceBuilder(
        model=model,
        tokenizer=tokenizer,
        reward_composer=composer,
        num_samples=pb_cfg.get("num_samples", 4),
        temperature=pb_cfg.get("temperature", 0.8),
        max_new_tokens=pb_cfg.get("max_new_tokens", 512),
    )

    prompts = [train_dataset[i]["input"] for i in range(len(train_dataset))]
    expected_answers = [train_dataset[i]["output"] for i in range(len(train_dataset))]
    thetas = [train_dataset[i].get("theta") for i in range(len(train_dataset))]

    preference_data = builder.build_dataset(
        prompts=prompts,
        expected_answers=expected_answers,
        thetas=thetas,
    )
    logger.info("Preference dataset size: %d pairs", len(preference_data))
    local_logger.log_metrics({"preference_pairs": len(preference_data)}, step=0)

    # ── Step 4: DPO Training ──────────────────────────────────────────────────
    from src.training.rlhf_trainer import RLHFTrainer

    trainer = RLHFTrainer(
        model=model,
        tokenizer=tokenizer,
        preference_data=preference_data,
        config=config,
        use_wandb=args.wandb,
        output_dir=str(output_dir),
    )
    checkpoint_path = trainer.train()

    logger.info("Training complete. Checkpoint: %s", checkpoint_path)
    local_logger.log_metrics({"checkpoint": checkpoint_path}, step=1)
    local_logger.close()

    if wandb_logger is not None:
        wandb_logger.finish()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    main()
    local_logger.close()

    if wandb_logger is not None:
        wandb_logger.finish()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    main()
