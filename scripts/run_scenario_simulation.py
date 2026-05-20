#!/usr/bin/env python
"""
Run bundled scenario simulation fixtures (mock/smoke by default). Vault seed: S5-02.

Example (dry-run, no network):
    python scripts/run_scenario_simulation.py --dry-run
    python scripts/run_scenario_simulation.py --scenario-type enterprise --dry-run

Live provider calls require ALLOW_LIVE_PROVIDER=1 (deferred to full sprint runs).
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from src.scenarios.simulation_runner import ScenarioSimulationRunner


def main() -> None:
    parser = argparse.ArgumentParser(description="Bundled scenario simulation runner")
    parser.add_argument(
        "--fixtures",
        default="data/eval/simulation_fixtures.json",
        help="Path to simulation fixtures JSON",
    )
    parser.add_argument(
        "--scenario-type",
        choices=["enterprise", "causal"],
        default=None,
        help="Filter fixtures by scenario type",
    )
    parser.add_argument("--fixture-id", action="append", default=None, help="Run specific fixture(s)")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--dry-run",
        action="store_true",
        default=True,
        help="Mock provider only (default)",
    )
    parser.add_argument("--live", action="store_true", help="Request live provider (gated)")
    parser.add_argument("--output", default=None, help="Optional JSON output path")
    args = parser.parse_args()

    runner = ScenarioSimulationRunner(
        fixtures_path=Path(args.fixtures),
        seed=args.seed,
        dry_run=args.dry_run or not args.live,
        live=args.live,
    )
    report = runner.run_all(
        scenario_type=args.scenario_type,
        fixture_ids=args.fixture_id,
    )

    if args.output:
        out = Path(args.output)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(report, indent=2), encoding="utf-8")

    print(json.dumps({"fixture_count": report["fixture_count"], "dry_run": report["dry_run"]}, indent=2))


if __name__ == "__main__":
    main()
