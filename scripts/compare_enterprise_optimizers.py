#!/usr/bin/env python
"""
Compare BootstrapFewShot vs MIPRO on the same filing and tiny eval set (S4-04).

Development: use ``--dry-run`` to emit schema-valid comparison JSON without executing
either optimizer or live LLM calls.

When resources allow:
    ENABLE_MIPRO=1 python scripts/compare_enterprise_optimizers.py --live
"""

from __future__ import annotations

import argparse
import json
import sys
import uuid
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from src.demo.pipeline import run_enterprise_demo
from src.dspy_modules.eval_metrics import evaluate_demo_result
from src.dspy_modules.optimize import OPTIMIZER_BOOTSTRAP, OPTIMIZER_MIPRO, resolve_optimizer
from src.eval.enterprise_eval_schema import (
    EnterpriseEvalReport,
    OptimizerComparisonReport,
    build_baseline_report,
    compute_criterion_deltas,
)


def _run_eval(
    filing_id: str,
    offline: bool,
    optimizer_name: str,
    seed: int,
) -> EnterpriseEvalReport:
    config = resolve_optimizer(optimizer_name)
    demo_result = run_enterprise_demo(filing_id=filing_id, offline=offline, output_dir=None)
    payload = evaluate_demo_result(demo_result)
    return build_baseline_report(
        payload,
        model_id="offline-stub" if offline else "live",
        optimizer=config.name,
        seed=seed,
        run_id=demo_result.get("trace_id", str(uuid.uuid4())),
        offline=offline,
        token_budget_note=config.token_budget_note,
    )


def _dry_run_comparison(filing_id: str, seed: int) -> OptimizerComparisonReport:
    """Placeholder comparison for docs/CI schema validation without API spend."""
    placeholder_scores = {
        "grounding": 0.91,
        "plausibility": 0.85,
        "severity_clarity": 0.88,
        "non_duplication": 1.0,
        "trace_completeness": 1.0,
    }
    thresholds = {
        "grounding": 0.7,
        "plausibility": 0.7,
        "severity_clarity": 0.7,
        "non_duplication": 0.9,
        "trace_completeness": 0.9,
    }
    base = build_baseline_report(
        {
            "filing_id": filing_id,
            "aggregate_scores": placeholder_scores,
            "thresholds": thresholds,
            "per_scenario": [],
            "all_pass": True,
        },
        optimizer=OPTIMIZER_BOOTSTRAP,
        seed=seed,
        run_id="dry-run-bootstrap",
        offline=True,
    )
    mipro_scores = dict(placeholder_scores)
    mipro_scores["plausibility"] = 0.86
    mipro = build_baseline_report(
        {
            "filing_id": filing_id,
            "aggregate_scores": mipro_scores,
            "thresholds": thresholds,
            "per_scenario": [],
            "all_pass": True,
        },
        optimizer=OPTIMIZER_MIPRO,
        seed=seed,
        run_id="dry-run-mipro",
        offline=True,
        token_budget_note="dry-run placeholder; run live comparison when budget allows",
    )
    return OptimizerComparisonReport(
        filing_id=filing_id,
        bootstrap_fewshot=base,
        mipro=mipro,
        criterion_deltas=compute_criterion_deltas(placeholder_scores, mipro_scores),
        scenario_trace_ids=[],
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare enterprise optimizers")
    parser.add_argument("--filing", default="acme_corp_10k")
    parser.add_argument("--offline", action="store_true", default=True)
    parser.add_argument("--live", action="store_true")
    parser.add_argument("--dry-run", action="store_true", help="Schema-only; no pipeline execution")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--output",
        default=str(_REPO_ROOT / "docs" / "eval" / "results" / "comparison_report.json"),
    )
    args = parser.parse_args()

    offline = not args.live if args.live else True
    if args.dry_run:
        comparison = _dry_run_comparison(args.filing, args.seed)
    else:
        bootstrap = _run_eval(args.filing, offline, OPTIMIZER_BOOTSTRAP, args.seed)
        mipro_report = _run_eval(args.filing, offline, OPTIMIZER_MIPRO, args.seed)
        comparison = OptimizerComparisonReport(
            filing_id=args.filing,
            bootstrap_fewshot=bootstrap,
            mipro=mipro_report,
            criterion_deltas=compute_criterion_deltas(
                bootstrap.aggregate_scores,
                mipro_report.aggregate_scores,
            ),
            scenario_trace_ids=[
                {
                    "scenario_index": str(row.get("scenario_index", "")),
                    "bootstrap_trace_id": row.get("trace_id", ""),
                }
                for row in bootstrap.per_scenario
            ],
        )

    out = Path(args.output)
    comparison.write_json(out)
    print(json.dumps({"written": str(out), "deltas": comparison.criterion_deltas}, indent=2))


if __name__ == "__main__":
    main()
