"""Manifest-driven artifact layout for training, eval, and demo exports."""

from __future__ import annotations

import json
import os
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


def trained_models_root() -> Path | None:
    """Canonical root: SCENARIO_TRAINED_MODELS_ROOT or unset."""
    env = os.environ.get("SCENARIO_TRAINED_MODELS_ROOT")
    if not env:
        return None
    return Path(env).expanduser().resolve()


def experiment_results_root() -> Path:
    return Path("experiments/results").resolve()


def enterprise_demo_root() -> Path:
    return Path("artifacts/enterprise_demo").resolve()


def distilled_data_root() -> Path:
    return Path("data/distilled").resolve()


def artifact_bundle_roots() -> list[Path]:
    """Canonical local roots mirrored by FTP and bucket sync."""
    roots = [
        experiment_results_root(),
        Path("experiments/adapters").resolve(),
        distilled_data_root(),
        Path("artifacts").resolve(),
        Path("docs/eval/results").resolve(),
        Path("benchmarks/results").resolve(),
    ]
    trained = trained_models_root()
    if trained is not None:
        roots.append(trained)
    return roots


@dataclass
class ArtifactManifest:
    """Structured record for train/eval artifact bundles."""

    run_id: str
    project: str = "scenario-reasoner-lm"
    model_id: str = ""
    adapter_path: str = ""
    eval_export_path: str = ""
    demo_artifact_path: str = ""
    commit_sha: str = ""
    created_at: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    metrics: dict[str, Any] = field(default_factory=dict)
    extra: dict[str, Any] = field(default_factory=dict)

    def write(self, root: Path | None = None) -> Path:
        root = root or experiment_results_root()
        root.mkdir(parents=True, exist_ok=True)
        out = root / self.run_id / "manifest.json"
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(asdict(self), indent=2), encoding="utf-8")
        return out

    @classmethod
    def load(cls, path: Path) -> ArtifactManifest:
        data = json.loads(path.read_text(encoding="utf-8"))
        return cls(**data)


def log_train_eval_manifest(
    *,
    run_id: str,
    model_id: str,
    metrics: dict[str, Any] | None = None,
    adapter_path: str | Path | None = None,
    eval_export_path: str | Path | None = None,
    demo_artifact_path: str | Path | None = None,
) -> Path:
    """Emit structured manifest next to experiment outputs."""
    manifest = ArtifactManifest(
        run_id=run_id,
        model_id=model_id,
        adapter_path=str(adapter_path) if adapter_path else "",
        eval_export_path=str(eval_export_path) if eval_export_path else "",
        demo_artifact_path=str(demo_artifact_path) if demo_artifact_path else "",
        commit_sha=os.environ.get("GIT_COMMIT_SHA", ""),
        metrics=metrics or {},
    )
    return manifest.write()
