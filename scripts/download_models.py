#!/usr/bin/env python
"""Download or verify Qwen portfolio models under SCENARIO_MODELS_ROOT."""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from src.models.model_registry import ModelRegistry

logger = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Cache Qwen portfolio models locally")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--model-id", type=str, help="Single registry model_id")
    group.add_argument("--tier", type=str, choices=["smoke", "mid", "large_gated"])
    group.add_argument("--all", action="store_true", help="All portfolio models")
    parser.add_argument(
        "--portfolio",
        default=str(_REPO_ROOT / "configs" / "models" / "qwen_portfolio.yaml"),
        help="Path to portfolio YAML",
    )
    parser.add_argument("--force", action="store_true", help="Re-download even if present")
    return parser.parse_args()


def _download_entry(registry: ModelRegistry, entry, force: bool) -> None:
    root = registry.require_models_root()
    dest = entry.local_dir(root)
    if dest.is_dir() and any(dest.iterdir()) and not force:
        logger.info("SKIP %s — already at %s", entry.model_id, dest)
        return

    try:
        from huggingface_hub import snapshot_download
    except ImportError as exc:
        raise ImportError("pip install huggingface_hub") from exc

    logger.info("Downloading %s (%s) -> %s", entry.model_id, entry.hub_id, dest)
    dest.mkdir(parents=True, exist_ok=True)
    snapshot_download(repo_id=entry.hub_id, local_dir=str(dest))


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s  %(message)s")
    args = parse_args()
    registry = ModelRegistry.load(Path(args.portfolio))

    if args.model_id:
        entries = [registry.get(args.model_id)]
    elif args.tier:
        entries = registry.get_by_tier(args.tier)
    else:
        entries = registry.all_entries()

    for entry in entries:
        _download_entry(registry, entry, args.force)

    logger.info("Done — %d model(s) processed", len(entries))


if __name__ == "__main__":
    main()
