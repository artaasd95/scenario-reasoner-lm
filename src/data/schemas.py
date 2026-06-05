"""Portfolio data schemas: preferences, feedback, scenario measurements (S9-DP-02)."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional


@dataclass
class PreferencePair:
    chosen: str
    rejected: str
    prompt: str
    theta: Dict[str, Any] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "PreferencePair":
        return cls(**{k: data[k] for k in cls.__dataclass_fields__ if k in data})


@dataclass
class FeedbackRecord:
  record_id: str
  coherence_score: float
  theta: Dict[str, Any] = field(default_factory=dict)
  notes: str = ""
  path_quality: Optional[float] = None

  def to_dict(self) -> Dict[str, Any]:
    return asdict(self)

  @classmethod
  def from_dict(cls, data: Dict[str, Any]) -> "FeedbackRecord":
    return cls(
      record_id=str(data["record_id"]),
      coherence_score=float(data["coherence_score"]),
      theta=dict(data.get("theta", {})),
      notes=str(data.get("notes", "")),
      path_quality=data.get("path_quality"),
    )


@dataclass
class ScenarioMeasurement:
  path_id: str
  feasibility: float
  tail_tag: str
  theta_stratum: str
  primary_quality_score: float
  theta: Dict[str, Any] = field(default_factory=dict)
  extra: Dict[str, Any] = field(default_factory=dict)

  def to_dict(self) -> Dict[str, Any]:
    return asdict(self)

  @classmethod
  def from_dict(cls, data: Dict[str, Any]) -> "ScenarioMeasurement":
    return cls(
      path_id=str(data["path_id"]),
      feasibility=float(data.get("feasibility", 0.0)),
      tail_tag=str(data.get("tail_tag", "none")),
      theta_stratum=str(data.get("theta_stratum", "")),
      primary_quality_score=float(data.get("primary_quality_score", 0.0)),
      theta=dict(data.get("theta", {})),
      extra=dict(data.get("extra", {})),
    )


def write_scenario_measurements(
    measurements: List[ScenarioMeasurement],
    path: Path,
) -> None:
  path.parent.mkdir(parents=True, exist_ok=True)
  payload = {"paths": [m.to_dict() for m in measurements]}
  path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
