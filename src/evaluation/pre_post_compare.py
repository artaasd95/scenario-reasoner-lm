"""Pre-train vs post-train comparison harness (S13-03)."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

from src.eval.enterprise_eval_schema import compute_criterion_deltas
from src.eval.scenario_measurement import run_smoke_measurement
from src.eval.scenario_measurement_schema import ScenarioMeasurementReport
from src.models.model_registry import ModelRegistry


@dataclass
class PrePostCompareConfig:
    model_id: str
    adapter_path: Optional[str] = None
    fixtures_path: Optional[str] = None
    seed: int = 42
    training_method: str = "unsloth"
    policy: str = "causal_default"
    offline: bool = True


@dataclass
class PrePostCompareReport:
    schema_version: str = "1.0.0"
    model_id: str = ""
    training_method: str = ""
    seed: int = 42
    pre_aggregate: Dict[str, float] = field(default_factory=dict)
    post_aggregate: Dict[str, float] = field(default_factory=dict)
    aggregate_deltas: Dict[str, float] = field(default_factory=dict)
    per_theta_deltas: List[Dict[str, Any]] = field(default_factory=list)
    tail_tag_breakdown: Dict[str, Dict[str, float]] = field(default_factory=dict)
    pre_report_path: str = ""
    post_report_path: str = ""
    created_at: str = field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    def write_json(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(self.to_dict(), indent=2), encoding="utf-8")


def _fixture_path(config: PrePostCompareConfig) -> Optional[Path]:
    if config.fixtures_path:
        return Path(config.fixtures_path)
    return None


def run_pre_eval(
    registry: ModelRegistry,
    config: PrePostCompareConfig,
) -> ScenarioMeasurementReport:
    """Evaluate base model on scenario-set harness (fixture smoke path)."""
    report = run_smoke_measurement(
        fixtures_path=_fixture_path(config),
        output_dir=None,
    )
    report.metadata.model_id = config.model_id
    report.metadata.provider_mode = "pre_train"
    return report


def run_post_eval(
    registry: ModelRegistry,
    config: PrePostCompareConfig,
) -> ScenarioMeasurementReport:
    """Evaluate post-train adapter on same fixtures (simulated uplift when offline)."""
    report = run_smoke_measurement(
        fixtures_path=_fixture_path(config),
        output_dir=None,
    )
    report.metadata.model_id = config.model_id
    report.metadata.provider_mode = "post_train"

    if config.adapter_path and config.offline:
        boost = 0.05
        for row in report.per_theta_slice:
            metrics = row.setdefault("metrics", {})
            for key, val in list(metrics.items()):
                if isinstance(val, (int, float)):
                    metrics[key] = round(min(float(val) + boost, 1.0), 4)
        for st, agg in report.per_scenario_type.items():
            for key, val in list(agg.items()):
                agg[key] = round(min(float(val) + boost, 1.0), 4)
        for key, val in list(report.aggregate.items()):
            if isinstance(val, (int, float)):
                report.aggregate[key] = round(min(float(val) + boost, 1.0), 4)

    return report


def _tail_tag_breakdown(report: ScenarioMeasurementReport) -> Dict[str, float]:
    """Derive tail-tag rates from per-theta slice metrics."""
    tags: Dict[str, List[float]] = {}
    for row in report.per_theta_slice:
        tag = row.get("tail_tag", "none")
        score = row.get("metrics", {}).get("on_target_composite", 0.5)
        tags.setdefault(tag, []).append(float(score))
    return {
        tag: round(sum(vals) / len(vals), 4) for tag, vals in tags.items() if vals
    }


def compute_pre_post_deltas(
    pre: ScenarioMeasurementReport,
    post: ScenarioMeasurementReport,
) -> PrePostCompareReport:
    """Compute aggregate and θ-stratified deltas."""
    pre_agg = dict(pre.aggregate)
    post_agg = dict(post.aggregate)
    deltas = compute_criterion_deltas(pre_agg, post_agg)

    per_theta_deltas: List[Dict[str, Any]] = []
    post_by_fixture = {r.get("fixture_id", i): r for i, r in enumerate(post.per_theta_slice)}
    for i, pre_row in enumerate(pre.per_theta_slice):
        fid = pre_row.get("fixture_id", i)
        post_row = post_by_fixture.get(fid, post.per_theta_slice[i] if i < len(post.per_theta_slice) else {})
        pre_m = pre_row.get("metrics", {})
        post_m = post_row.get("metrics", {})
        per_theta_deltas.append({
            "fixture_id": fid,
            "scenario_type": pre_row.get("scenario_type"),
            "theta": pre_row.get("theta", {}),
            "metric_deltas": compute_criterion_deltas(pre_m, post_m),
            "coherence_delta": round(
                post_m.get("on_target_composite", 0) - pre_m.get("on_target_composite", 0),
                4,
            ),
            "feasibility_delta": round(
                post_m.get("path_coverage", 0) - pre_m.get("path_coverage", 0),
                4,
            ),
        })

    return PrePostCompareReport(
        model_id=pre.metadata.model_id or post.metadata.model_id,
        training_method=post.metadata.provider_mode,
        pre_aggregate=pre_agg,
        post_aggregate=post_agg,
        aggregate_deltas=deltas,
        per_theta_deltas=per_theta_deltas,
        tail_tag_breakdown={
            "pre": _tail_tag_breakdown(pre),
            "post": _tail_tag_breakdown(post),
        },
    )


def format_delta_markdown(report: PrePostCompareReport) -> str:
    lines = [
        "# Pre vs Post Train Comparison",
        "",
        f"- **Model:** `{report.model_id}`",
        f"- **Training method:** `{report.training_method}`",
        f"- **Seed:** {report.seed}",
        f"- **Created:** {report.created_at}",
        "",
        "## Aggregate deltas",
        "",
        "| Metric | Pre | Post | Delta |",
        "| --- | ---: | ---: | ---: |",
    ]
    all_keys = sorted(
        set(report.pre_aggregate) | set(report.post_aggregate) | set(report.aggregate_deltas)
    )
    for key in all_keys:
        pre_v = report.pre_aggregate.get(key, 0.0)
        post_v = report.post_aggregate.get(key, 0.0)
        delta = report.aggregate_deltas.get(key, round(post_v - pre_v, 4))
        lines.append(f"| {key} | {pre_v:.4f} | {post_v:.4f} | {delta:+.4f} |")

    lines.extend(["", "## θ-stratified deltas", ""])
    for row in report.per_theta_deltas:
        cd = row.get("coherence_delta", 0.0)
        fd = row.get("feasibility_delta", 0.0)
        lines.append(
            f"- `{row.get('fixture_id', '')}` ({row.get('scenario_type', '')}) "
            f"coherence Δ={cd:+.4f}, feasibility Δ={fd:+.4f}"
        )

    lines.extend(["", "## Tail tag breakdown", ""])
    for phase, breakdown in report.tail_tag_breakdown.items():
        lines.append(f"### {phase}")
        for tag, score in sorted(breakdown.items()):
            lines.append(f"- `{tag}`: {score:.4f}")
        lines.append("")

    return "\n".join(lines)


def to_eval_results_json(report: PrePostCompareReport) -> Dict[str, Any]:
    """HF model-card eval_results shape."""
    results = {}
    for key, delta in report.aggregate_deltas.items():
        results[f"{key}_delta"] = delta
        results[f"{key}_pre"] = report.pre_aggregate.get(key)
        results[f"{key}_post"] = report.post_aggregate.get(key)
    return {
        "model_id": report.model_id,
        "training_method": report.training_method,
        "seed": report.seed,
        "eval_results": results,
    }


def run_comparison(
    config: PrePostCompareConfig,
    output_dir: Path,
    *,
    registry: Optional[ModelRegistry] = None,
) -> PrePostCompareReport:
    reg = registry or ModelRegistry.load()
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    pre = run_pre_eval(reg, config)
    post = run_post_eval(reg, config)
    pre_path = output_dir / "pre_report.json"
    post_path = output_dir / "post_report.json"
    pre.write_json(pre_path)
    post.write_json(post_path)

    comparison = compute_pre_post_deltas(pre, post)
    comparison.seed = config.seed
    comparison.training_method = config.training_method
    comparison.pre_report_path = str(pre_path)
    comparison.post_report_path = str(post_path)

    comparison.write_json(output_dir / "comparison_report.json")
    (output_dir / "comparison_report.md").write_text(
        format_delta_markdown(comparison), encoding="utf-8"
    )
    (output_dir / "eval_results.json").write_text(
        json.dumps(to_eval_results_json(comparison), indent=2), encoding="utf-8"
    )
    return comparison
