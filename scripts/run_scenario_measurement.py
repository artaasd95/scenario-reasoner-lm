#!/usr/bin/env python
"""
Multi-scenario measurement harness (smoke/fixtures only by default). Vault seed: S5-04.

Example:
    python scripts/run_scenario_measurement.py --smoke --output docs/eval/results/scenario_measurement
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from src.eval.scenario_measurement import run_smoke_measurement
from src.scenarios.resource_gate import assert_mock_or_gated


def main() -> None:
    parser = argparse.ArgumentParser(description="Scenario measurement harness")
    parser.add_argument(
        "--smoke",
        action="store_true",
        default=True,
        help="Bundled fixtures only (default)",
    )
    parser.add_argument("--live", action="store_true", help="Live eval (gated; deferred)")
    parser.add_argument(
        "--output",
        default="docs/eval/results/scenario_measurement",
        help="Output directory for measurement_report.json",
    )
    parser.add_argument(
        "--fixtures",
        default="data/eval/simulation_fixtures.json",
        help="Simulation fixtures path",
    )
    args = parser.parse_args()

    if args.live:
        assert_mock_or_gated(live_requested=True)

    report = run_smoke_measurement(
        fixtures_path=Path(args.fixtures),
        output_dir=Path(args.output) if args.smoke or not args.live else None,
    )
    summary = {
        "schema_version": report.schema_version,
        "smoke_mode": report.metadata.smoke_mode,
        "aggregate": report.aggregate,
        "theta_slice_count": len(report.per_theta_slice),
    }
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
