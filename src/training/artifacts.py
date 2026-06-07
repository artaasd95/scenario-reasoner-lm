"""Post-train artifact promotion to SCENARIO_TRAINED_MODELS_ROOT."""

from __future__ import annotations

import json
import logging
import shutil
from pathlib import Path
from typing import Any, Dict, Optional

from src.models.model_registry import ModelRegistry

logger = logging.getLogger(__name__)


def promote_checkpoint(
    checkpoint_path: str,
    config: Dict[str, Any],
    *,
    registry: Optional[ModelRegistry] = None,
) -> Optional[str]:
    """
    Copy DPO checkpoint to ``{SCENARIO_TRAINED_MODELS_ROOT}/{model_id}/{run_id}/``.

    Returns destination path, or None if env var is unset.
    """
    trained_root = ModelRegistry.trained_root()
    if trained_root is None:
        logger.info(
            "SCENARIO_TRAINED_MODELS_ROOT not set — skipping artifact promotion"
        )
        return None

    model_id = config.get("model_id") or config.get("model_name_or_path", "unknown")
    run_id = config.get("experiment_name", "run")
    dest = trained_root / str(model_id) / str(run_id)
    dest.mkdir(parents=True, exist_ok=True)

    src = Path(checkpoint_path)
    for item in src.iterdir():
        target = dest / item.name
        if item.is_dir():
            if target.exists():
                shutil.rmtree(target)
            shutil.copytree(item, target)
        else:
            shutil.copy2(item, target)

    meta = {
        "model_id": model_id,
        "hub_id": config.get("hub_id"),
        "backend": config.get("training", {}).get("backend", "trl"),
        "policy": config.get("training", {}).get("policy"),
        "seed": config.get("data", {}).get("seed"),
        "base_model_path": config.get("model_name_or_path"),
        "checkpoint_source": str(src),
    }
    (dest / "train_config.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    logger.info("Promoted checkpoint to %s", dest)
    return str(dest)
