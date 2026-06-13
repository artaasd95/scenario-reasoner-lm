"""Export training checkpoints into a shared handoff format."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch


@dataclass
class ExportResult:
    checkpoint_path: Path
    export_dir: Path
    manifest_path: Path


class CheckpointExporter:
    """Create RADA/raft-compatible checkpoint export bundles."""

    def __init__(self, output_root: str | Path) -> None:
        self._output_root = Path(output_root)

    def export_for_rada(
        self,
        checkpoint_path: str | Path,
        *,
        adapter_config: dict[str, Any] | None = None,
        model_id: str | None = None,
    ) -> ExportResult:
        source = Path(checkpoint_path)
        if not source.exists():
            raise FileNotFoundError(f"checkpoint not found: {source}")

        payload = torch.load(source, map_location="cpu", weights_only=False)
        export_dir = self._output_root / source.stem
        export_dir.mkdir(parents=True, exist_ok=True)

        out_ckpt = export_dir / "model_checkpoint.pt"
        torch.save(payload, out_ckpt)

        manifest = {
            "format": "rada_checkpoint_export_v1",
            "checkpoint_file": out_ckpt.name,
            "source_checkpoint": str(source),
            "model_id": model_id or str(payload.get("model_id", "unknown")),
            "adapter_config": adapter_config or {
                "backend": payload.get("backend", "trl"),
                "weights_key": "model_state_dict",
            },
        }
        manifest_path = export_dir / "export_manifest.json"
        manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
        return ExportResult(out_ckpt, export_dir, manifest_path)
