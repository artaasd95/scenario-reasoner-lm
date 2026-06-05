"""YAML pipeline configuration loader (S9-DP-03)."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

import yaml


@dataclass
class PipelineConfig:
    mode: str  # train | eval
    source: Dict[str, Any] = field(default_factory=dict)
    stages: List[str] = field(default_factory=lambda: ["normalize", "measure", "label", "split", "filter"])
    output_dir: str = "data/processed"
    split: Dict[str, float] = field(default_factory=lambda: {"train": 0.9, "val": 0.1})
    filter: Dict[str, Any] = field(default_factory=lambda: {"drop_infeasible": True})
    measure: Dict[str, Any] = field(default_factory=dict)
    label: Dict[str, Any] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "PipelineConfig":
        return cls(
            mode=str(data.get("mode", "train")),
            source=dict(data.get("source", {})),
            stages=list(data.get("stages", ["normalize", "measure", "label", "split", "filter"])),
            output_dir=str(data.get("output_dir", "data/processed")),
            split=dict(data.get("split", {"train": 0.9, "val": 0.1})),
            filter=dict(data.get("filter", {"drop_infeasible": True})),
            measure=dict(data.get("measure", {})),
            label=dict(data.get("label", {})),
            metadata=dict(data.get("metadata", {})),
        )


def load_pipeline_config(path: Path | str) -> PipelineConfig:
    config_path = Path(path)
    if not config_path.is_file():
        raise FileNotFoundError(f"Pipeline config not found: {config_path}")
    data = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    return PipelineConfig.from_dict(data or {})
