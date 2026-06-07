"""Load training configs from JSON or YAML."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict


def load_training_config(path: str | Path) -> Dict[str, Any]:
    p = Path(path)
    with open(p, encoding="utf-8") as f:
        if p.suffix.lower() in (".yaml", ".yml"):
            import yaml

            return yaml.safe_load(f) or {}
        return json.load(f)
