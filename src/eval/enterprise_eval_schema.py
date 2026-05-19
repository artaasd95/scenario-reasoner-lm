"""
Stable JSON schema for enterprise eval and optimizer comparison artifacts.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional


SCHEMA_VERSION = "1.0.0"


@dataclass
class EvalRunMetadata:
    model_id: str
    optimizer: str
    seed: int
    run_id: str
    filing_id: str = "acme_corp_10k"
    eval_set_path: str = "data/eval/enterprise_risk_tiny.jsonl"
    offline: bool = True
    smoke_mode: bool = False
    token_budget_note: str = ""


@dataclass
class EnterpriseEvalReport:
    """BootstrapFewShot or single-optimizer eval artifact."""

    schema_version: str = SCHEMA_VERSION
    metadata: EvalRunMetadata = field(default_factory=lambda: EvalRunMetadata(
        model_id="offline-stub",
        optimizer="BootstrapFewShot",
        seed=42,
        run_id="",
    ))
    aggregate_scores: Dict[str, float] = field(default_factory=dict)
    thresholds: Dict[str, float] = field(default_factory=dict)
    per_scenario: List[Dict[str, Any]] = field(default_factory=list)
    all_pass: bool = False
    created_at: str = field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        return d

    def write_json(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(self.to_dict(), indent=2), encoding="utf-8")

    def write_markdown(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(format_eval_markdown(self), encoding="utf-8")


@dataclass
class OptimizerComparisonReport:
    """Side-by-side BootstrapFewShot vs MIPRO on the same filing and eval set."""

    schema_version: str = SCHEMA_VERSION
    filing_id: str = "acme_corp_10k"
    eval_set_path: str = "data/eval/enterprise_risk_tiny.jsonl"
    bootstrap_fewshot: Optional[EnterpriseEvalReport] = None
    mipro: Optional[EnterpriseEvalReport] = None
    criterion_deltas: Dict[str, float] = field(default_factory=dict)
    scenario_trace_ids: List[Dict[str, str]] = field(default_factory=list)
    created_at: str = field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )

    def to_dict(self) -> Dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "filing_id": self.filing_id,
            "eval_set_path": self.eval_set_path,
            "bootstrap_fewshot": self.bootstrap_fewshot.to_dict() if self.bootstrap_fewshot else None,
            "mipro": self.mipro.to_dict() if self.mipro else None,
            "criterion_deltas": self.criterion_deltas,
            "scenario_trace_ids": self.scenario_trace_ids,
            "created_at": self.created_at,
        }

    def write_json(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(self.to_dict(), indent=2), encoding="utf-8")


def build_baseline_report(
    eval_payload: dict,
    *,
    model_id: str = "offline-stub",
    optimizer: str = "BootstrapFewShot",
    seed: int = 42,
    run_id: str = "",
    offline: bool = True,
    smoke_mode: bool = False,
    token_budget_note: str = "",
) -> EnterpriseEvalReport:
    meta = EvalRunMetadata(
        model_id=model_id,
        optimizer=optimizer,
        seed=seed,
        run_id=run_id or eval_payload.get("run_trace_id", ""),
        filing_id=eval_payload.get("filing_id", "acme_corp_10k"),
        offline=offline,
        smoke_mode=smoke_mode,
        token_budget_note=token_budget_note,
    )
    return EnterpriseEvalReport(
        metadata=meta,
        aggregate_scores=eval_payload.get("aggregate_scores", {}),
        thresholds=eval_payload.get("thresholds", {}),
        per_scenario=eval_payload.get("per_scenario", []),
        all_pass=bool(eval_payload.get("all_pass", False)),
    )


def compute_criterion_deltas(
    baseline: Dict[str, float],
    candidate: Dict[str, float],
) -> Dict[str, float]:
    keys = set(baseline) | set(candidate)
    return {k: round(candidate.get(k, 0.0) - baseline.get(k, 0.0), 4) for k in sorted(keys)}


def format_eval_markdown(report: EnterpriseEvalReport) -> str:
    meta = report.metadata
    lines = [
        "# Enterprise Risk Eval Report",
        "",
        f"- **Schema:** {report.schema_version}",
        f"- **Created:** {report.created_at}",
        f"- **Model:** `{meta.model_id}`",
        f"- **Optimizer:** `{meta.optimizer}`",
        f"- **Seed:** {meta.seed}",
        f"- **Run ID:** `{meta.run_id}`",
        f"- **Filing:** `{meta.filing_id}`",
        f"- **Offline:** {meta.offline}",
        f"- **All pass:** {report.all_pass}",
        "",
    ]
    if meta.token_budget_note:
        lines.extend([f"- **Budget note:** {meta.token_budget_note}", ""])

    lines.extend(["## Aggregate scores (S2 criteria)", "", "| Criterion | Score | Threshold |", "| --- | ---: | ---: |"])
    for name, score in sorted(report.aggregate_scores.items()):
        thresh = report.thresholds.get(name, "")
        lines.append(f"| {name} | {score} | {thresh} |")
    lines.append("")

    if report.per_scenario:
        lines.extend(["## Per-scenario", ""])
        for row in report.per_scenario:
            lines.append(
                f"- **#{row.get('scenario_index')}** `{row.get('trace_id', '')}` "
                f"— {row.get('title', '')} — pass={row.get('pass')}"
            )
        lines.append("")

    lines.append(
        "_Reserved for development: re-run `scripts/run_enterprise_eval.py` when API budget allows._"
    )
    return "\n".join(lines)
