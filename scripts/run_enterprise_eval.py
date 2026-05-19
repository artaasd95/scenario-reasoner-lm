#!/usr/bin/env python
"""
Score enterprise risk demo output on the tiny eval set (S4-01).

Writes JSON + Markdown under ``docs/eval/results/<optimizer_slug>/``.

Development default: ``--offline`` (no paid API). Use ``--skip-pipeline`` with
``--from-result`` to re-score an existing ``demo_result.json`` without re-running
the pipeline.

Example (when resources allow — not required in dev scaffolding):
    python scripts/run_enterprise_eval.py --offline
    python scripts/run_enterprise_eval.py --from-result artifacts/enterprise_demo/demo_result.json
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import uuid
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from src.demo.pipeline import run_enterprise_demo
from src.dspy_modules.eval_metrics import evaluate_demo_result, load_enterprise_eval_set
from src.eval.enterprise_eval_schema import build_baseline_report


def _default_model_id(offline: bool) -> str:
    if offline:
        return "offline-stub"
    return os.getenv("ENTERPRISE_MODEL_NAME", "gpt-4o-mini")


def main() -> None:
    parser = argparse.ArgumentParser(description="Enterprise risk tiny eval (S2 criteria)")
    parser.add_argument("--filing", default="acme_corp_10k")
    parser.add_argument("--offline", action="store_true", help="Offline stubs (default in dev)")
    parser.add_argument("--live", action="store_true", help="Live DSPy + LLM (consumes API budget)")
    parser.add_argument(
        "--optimizer",
        default="BootstrapFewShot",
        choices=["BootstrapFewShot", "MIPRO"],
        help="Optimizer label recorded in report metadata",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--run-id", default="", help="Override run id in report")
    parser.add_argument(
        "--output-dir",
        default="",
        help="Defaults to docs/eval/results/<optimizer_slug>/",
    )
    parser.add_argument(
        "--from-result",
        default="",
        help="Score existing demo_result.json instead of running pipeline",
    )
    parser.add_argument(
        "--skip-pipeline",
        action="store_true",
        help="Alias: require --from-result",
    )
    parser.add_argument(
        "--eval-set",
        default=str(_REPO_ROOT / "data" / "eval" / "enterprise_risk_tiny.jsonl"),
    )
    parser.add_argument(
        "--token-budget-note",
        default="BootstrapFewShot baseline: max_bootstrapped_demos=4, max_labeled_demos=8 (see optimize.py)",
    )
    args = parser.parse_args()

    offline = True
    if args.live:
        offline = False
    if args.offline:
        offline = True

    eval_records = load_enterprise_eval_set(args.eval_set)

    if args.from_result or args.skip_pipeline:
        result_path = Path(args.from_result)
        if not result_path.is_file():
            print(f"Missing --from-result file: {result_path}", file=sys.stderr)
            sys.exit(1)
        demo_result = json.loads(result_path.read_text(encoding="utf-8"))
    else:
        demo_result = run_enterprise_demo(
            filing_id=args.filing,
            offline=offline,
            output_dir=None,
        )

    eval_payload = evaluate_demo_result(demo_result, eval_records)
    run_id = args.run_id or demo_result.get("trace_id") or str(uuid.uuid4())

    report = build_baseline_report(
        eval_payload,
        model_id=_default_model_id(offline),
        optimizer=args.optimizer,
        seed=args.seed,
        run_id=run_id,
        offline=offline,
        token_budget_note=args.token_budget_note,
    )

    slug = args.optimizer.replace(" ", "_").lower()
    out_dir = Path(args.output_dir) if args.output_dir else _REPO_ROOT / "docs" / "eval" / "results" / slug
    out_dir.mkdir(parents=True, exist_ok=True)

    json_path = out_dir / "baseline_report.json"
    md_path = out_dir / "baseline_report.md"
    report.write_json(json_path)
    report.write_markdown(md_path)

    print(
        json.dumps(
            {
                "written_json": str(json_path),
                "written_md": str(md_path),
                "aggregate_scores": report.aggregate_scores,
                "all_pass": report.all_pass,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
