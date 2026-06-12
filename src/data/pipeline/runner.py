"""Pipeline stage runner: normalize → measure → label → split → filter (S9-DP-03)."""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

from src.data.cards.scenario_path_row import ScenarioPathRow
from src.data.pipeline.config import PipelineConfig
from src.data.schemas import ScenarioMeasurement, write_scenario_measurements
from src.data.sources.base import DataSource
from src.data.theta_normalization import normalize_theta_record
from src.scenarios.feasibility import assess_path_feasibility
from src.metrics.tail_scenario_metrics import tag_tail_scenario

logger = logging.getLogger(__name__)


@dataclass
class PipelineRunStats:
    loaded: int = 0
    normalized: int = 0
    measured: int = 0
    labeled: int = 0
    dropped: int = 0
    enriched: int = 0
    stage_counts: Dict[str, int] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "loaded": self.loaded,
            "normalized": self.normalized,
            "measured": self.measured,
            "labeled": self.labeled,
            "dropped": self.dropped,
            "enriched": self.enriched,
            "stage_counts": dict(self.stage_counts),
        }


class PipelineRunner:
    def __init__(self, config: PipelineConfig, source: Optional[DataSource] = None) -> None:
        self.config = config
        self._source = source

    def run(self, source: Optional[DataSource] = None) -> Dict[str, Any]:
        src = source or self._source
        if src is None:
            raise ValueError("PipelineRunner requires a DataSource")

        stats = PipelineRunStats()
        rows = list(src.iter_rows())
        stats.loaded = len(rows)
        artifacts: List[Dict[str, Any]] = []

        for raw in rows:
            row = self._normalize_row(raw, stats)
            if row is None:
                continue
            measured = self._measure_row(row, stats)
            labeled = self._label_row(measured, stats)
            artifacts.append(labeled)

        if "filter" in self.config.stages:
            artifacts, filter_dropped = self._filter_rows(artifacts)
            stats.dropped += filter_dropped

        if "split" in self.config.stages:
            artifacts = self._split_rows(artifacts)

        out_dir = Path(self.config.output_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        out_path = out_dir / f"pipeline_{self.config.mode}.jsonl"
        with out_path.open("w", encoding="utf-8") as fh:
            for row in artifacts:
                fh.write(json.dumps(row) + "\n")

        if self.config.mode == "eval":
            measurements = [
                ScenarioMeasurement.from_dict(r["measurement"])
                for r in artifacts
                if "measurement" in r
            ]
            write_scenario_measurements(measurements, out_dir / "scenario_measurement.json")

        return {"output": str(out_path), "stats": stats.to_dict()}

    def _normalize_row(self, raw: Dict[str, Any], stats: PipelineRunStats) -> Optional[Dict[str, Any]]:
        if "normalize" not in self.config.stages:
            return raw
        theta = raw.get("theta", {})
        kind = raw.get("theta_kind")
        result = normalize_theta_record(theta, kind=kind)
        if not result.ok:
            stats.dropped += 1
            return None
        stats.normalized += 1
        stats.stage_counts["normalize"] = stats.normalized
        path_row = ScenarioPathRow.from_dict(raw) if "path_id" in raw else None
        out = dict(raw)
        out["theta"] = result.theta.to_dict() if hasattr(result.theta, "to_dict") else theta
        out["theta_kind"] = result.kind
        if path_row is not None:
            out["path_row"] = path_row.to_dict()
        return out

    def _measure_row(self, row: Dict[str, Any], stats: PipelineRunStats) -> Dict[str, Any]:
        if "measure" not in self.config.stages:
            return row
        feasibility = assess_path_feasibility(row)
        tail = tag_tail_scenario(row)
        primary = float(row.get("primary_quality_score", feasibility.get("score", 0.5)))
        stratum = row.get("theta_stratum") or row.get("theta_kind", "unknown")
        measurement = ScenarioMeasurement(
            path_id=str(row.get("path_id", row.get("fixture_id", "path"))),
            feasibility=feasibility["feasible_score"],
            tail_tag=tail["tail_tag"],
            theta_stratum=str(stratum),
            primary_quality_score=primary,
            theta=dict(row.get("theta", {})),
        )
        row["measurement"] = measurement.to_dict()
        stats.measured += 1
        stats.stage_counts["measure"] = stats.measured
        return row

    def _label_row(self, row: Dict[str, Any], stats: PipelineRunStats) -> Dict[str, Any]:
        if "label" not in self.config.stages:
            return row
        label_cfg = self.config.label
        row["path_quality"] = float(row.get("path_quality", row.get("measurement", {}).get("primary_quality_score", 0.5)))
        row["chosen_rejected"] = label_cfg.get("chosen_rejected", row.get("chosen_rejected", "chosen"))
        stats.labeled += 1
        stats.stage_counts["label"] = stats.labeled
        return row

    def _filter_rows(self, rows: List[Dict[str, Any]]) -> tuple[List[Dict[str, Any]], int]:
        if not self.config.filter.get("drop_infeasible", True):
            return rows, 0
        kept: List[Dict[str, Any]] = []
        dropped = 0
        threshold = float(self.config.filter.get("feasibility_threshold", 0.2))
        for row in rows:
            feas = row.get("measurement", {}).get("feasibility", 1.0)
            if feas < threshold:
                dropped += 1
                continue
            kept.append(row)
        return kept, dropped

    def _split_rows(self, rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        train_frac = float(self.config.split.get("train", 0.9))
        n_train = int(len(rows) * train_frac)
        for i, row in enumerate(rows):
            row["split"] = "train" if i < n_train else "val"
        return rows
