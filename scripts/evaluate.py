"""
Evaluation entry point scaffold for Scenario Reasoner LM.

Extend this script with concrete model loading and inference logic.
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

logger = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Scenario Reasoner LM — Evaluation Script")
    parser.add_argument("--config", type=str, required=True, help="Path to JSON config file")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to model checkpoint")
    parser.add_argument("--output-dir", type=str, default="./eval_outputs", help="Output directory")
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
    from src.metrics.base_metrics import MetricRegistry
    from src.monitoring.cot_monitor import CoTMonitor
    from src.monitoring.tot_monitor import ToTMonitor
    from src.monitoring.aha_monitor import AhaMonitor

    local_logger = LocalLogger(
        name=config.get("experiment_name", "eval"),
        log_dir=str(output_dir / "logs"),
    )
    local_logger.log_config({**config, "checkpoint": args.checkpoint})

    cot_monitor = CoTMonitor()
    tot_monitor = ToTMonitor()
    aha_monitor = AhaMonitor()

    # --------------------------------------------------------------------------
    # TODO: implement the following:
    #   1. Load dataset via ScenarioDataset or HFDatasetWrapper
    #   2. Load model from checkpoint
    #   3. Create DataLoader (shuffle=False)
    #   4. Inference loop:
    #      a. Forward pass (torch.no_grad())
    #      b. Decode output tokens
    #      c. Update MetricRegistry
    #      d. Update monitors: cot_monitor.update(outputs), etc.
    #   5. Compute final metrics
    #   6. Log and save results
    # --------------------------------------------------------------------------

    logger.info("Evaluation scaffold ready. Implement model and inference loop above.")
    local_logger.close()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    main()
