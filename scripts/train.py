"""
Training entry point scaffold for Scenario Reasoner LM.

This script provides a minimal training loop skeleton.  Extend it with
concrete model, tokeniser, optimizer, and scheduler implementations.
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
    parser.add_argument("--output-dir", type=str, default="./outputs", help="Output directory")
    parser.add_argument("--wandb", action="store_true", help="Enable Weights & Biases logging")
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
    from src.logging.wandb_logger import WandbLogger

    local_logger = LocalLogger(
        name=config.get("experiment_name", "experiment"),
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

    from src.monitoring.cot_monitor import CoTMonitor
    from src.monitoring.tot_monitor import ToTMonitor
    from src.monitoring.aha_monitor import AhaMonitor

    cot_monitor = CoTMonitor()
    tot_monitor = ToTMonitor()
    aha_monitor = AhaMonitor()

    # --------------------------------------------------------------------------
    # TODO: implement the following:
    #   1. Load dataset via ScenarioDataset or HFDatasetWrapper
    #   2. Instantiate model and tokeniser
    #   3. Create DataLoader
    #   4. Create optimiser and scheduler
    #   5. Training loop (for each epoch, for each batch):
    #      a. Forward pass
    #      b. Compute loss with custom BaseLoss subclass
    #      c. Backward pass + gradient step
    #      d. Update metrics via MetricRegistry
    #      e. Update monitors: cot_monitor.update(outputs), etc.
    #      f. Log: local_logger.log_step(step, metrics)
    #   6. Evaluation loop
    #   7. Checkpoint saving
    # --------------------------------------------------------------------------

    logger.info("Training scaffold ready. Implement model and training loop above.")
    local_logger.close()

    if wandb_logger is not None:
        wandb_logger.finish()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    main()
