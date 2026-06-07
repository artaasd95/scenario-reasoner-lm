#!/usr/bin/env python
"""Compare base (pre-train) vs LoRA adapter (post-train) on scenario harness."""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from src.evaluation.pre_post_compare import PrePostCompareConfig, run_comparison

logger = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Pre vs post train comparison")
    parser.add_argument("--model-id", required=True, help="Registry model_id")
    parser.add_argument("--adapter-path", default=None, help="LoRA adapter checkpoint dir")
    parser.add_argument(
        "--fixtures",
        default="data/eval/simulation_fixtures.json",
        help="Scenario fixtures path",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--training-method", default="unsloth")
    parser.add_argument("--policy", default="causal_default")
    parser.add_argument("--output", required=True, help="Output directory")
    parser.add_argument("--mini-train", action="store_true", help="Reserved for integration hook")
    return parser.parse_args()


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s  %(message)s")
    args = parse_args()

    if args.mini_train:
        logger.info("mini-train hook acknowledged (use train.py for full DPO)")

    config = PrePostCompareConfig(
        model_id=args.model_id,
        adapter_path=args.adapter_path,
        fixtures_path=args.fixtures,
        seed=args.seed,
        training_method=args.training_method,
        policy=args.policy,
        offline=True,
    )
    report = run_comparison(config, Path(args.output))
    logger.info(
        "Comparison complete — %d aggregate deltas written to %s",
        len(report.aggregate_deltas),
        args.output,
    )


if __name__ == "__main__":
    main()
