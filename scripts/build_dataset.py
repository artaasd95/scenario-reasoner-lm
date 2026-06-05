#!/usr/bin/env python3
"""
Build datasets from data-platform YAML config (S9-DP-05).

Usage::

    python scripts/build_dataset.py --config configs/data/train.yaml
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.data.pipeline.config import load_pipeline_config
from src.data.pipeline.runner import PipelineRunner
from src.data.pipeline.source_factory import build_source

logger = logging.getLogger(__name__)


def main() -> None:
    parser = argparse.ArgumentParser(description="Build dataset from pipeline config")
    parser.add_argument("--config", type=str, required=True, help="YAML config path")
    args = parser.parse_args()

    config = load_pipeline_config(args.config)
    source = build_source(config.source)
    runner = PipelineRunner(config, source=source)
    result = runner.run()

    logger.info("Pipeline complete: %s", json.dumps(result["stats"], indent=2))
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s  %(message)s")
    main()
