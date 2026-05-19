#!/usr/bin/env python
"""
Run the bundled enterprise-risk 10-K demo (offline by default).

Example:
    python scripts/run_enterprise_demo.py --offline
    python scripts/run_enterprise_demo.py --filing acme_corp_10k --output artifacts/demo
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from src.demo.pipeline import run_enterprise_demo
from src.dspy_modules.optimize import OPTIMIZER_BOOTSTRAP, OPTIMIZER_MIPRO, resolve_optimizer


def main() -> None:
    parser = argparse.ArgumentParser(description="Enterprise risk 10-K demo")
    parser.add_argument("--filing", default="acme_corp_10k", help="Bundled filing id or file path")
    parser.add_argument("--offline", action="store_true", help="Use offline stubs")
    parser.add_argument("--live", action="store_true", help="Use live DSPy + LLM")
    parser.add_argument(
        "--optimizer",
        default=OPTIMIZER_BOOTSTRAP,
        choices=[OPTIMIZER_BOOTSTRAP, OPTIMIZER_MIPRO],
        help="DSPy optimizer label (MIPRO requires ENABLE_MIPRO=1)",
    )
    parser.add_argument("--seed", type=int, default=42, help="Optimizer seed")
    parser.add_argument("--output", default="artifacts/enterprise_demo")
    args = parser.parse_args()

    offline = not args.live if args.live else True
    if args.offline:
        offline = True

    optimizer_config = resolve_optimizer(args.optimizer, seed=args.seed)

    result = run_enterprise_demo(
        filing_id=args.filing,
        offline=offline,
        output_dir=args.output,
        optimizer_config=optimizer_config,
    )
    print(json.dumps({"trace_id": result["trace_id"], "scenarios": len(result["scenarios"])}, indent=2))


if __name__ == "__main__":
    main()
