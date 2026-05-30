#!/usr/bin/env python
"""
Scenario-set reasoning eval harness (smoke/fixtures only by default). Vault seed: S6-05 / S7-05.

Example:
    python scripts/run_scenario_reasoning_eval.py --smoke --output docs/eval/results/scenario_reasoning
    EXECUTION_SPRINT_GATE=1 python scripts/run_scenario_reasoning_eval.py --full
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from src.eval.scenario_reasoning_eval import run_full_reasoning_eval, run_smoke_reasoning_eval
from src.scenarios.resource_gate import assert_execution_sprint_allowed, assert_mock_or_gated


def main() -> None:
    parser = argparse.ArgumentParser(description="Scenario-set reasoning eval harness")
    parser.add_argument(
        "--smoke",
        action="store_true",
        default=True,
        help="Bundled fixtures only (default)",
    )
    parser.add_argument(
        "--full",
        action="store_true",
        help="Full pipeline (requires EXECUTION_SPRINT_GATE=1)",
    )
    parser.add_argument("--live", action="store_true", help="Live provider (gated; deferred)")
    parser.add_argument(
        "--output",
        default="docs/eval/results/scenario_reasoning",
        help="Output directory for reasoning_eval_report.json",
    )
    parser.add_argument("--seed", type=int, default=42, help="RNG seed for path generators")
    args = parser.parse_args()

    if args.live:
        assert_mock_or_gated(live_requested=True)

    if args.full:
        assert_execution_sprint_allowed(full_pipeline=True)
        report = run_full_reasoning_eval(
            output_dir=Path(args.output),
            seed=args.seed,
            live=args.live,
        )
    else:
        report = run_smoke_reasoning_eval(
            output_dir=Path(args.output),
            seed=args.seed,
        )

    summary = {
        "schema_version": report.schema_version,
        "smoke_mode": report.metadata.smoke_mode,
        "execution_sprint": report.metadata.execution_sprint,
        "per_path_type": report.per_path_type,
        "aggregate": report.aggregate,
        "theta_slice_count": len(report.per_theta_slice),
    }
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
