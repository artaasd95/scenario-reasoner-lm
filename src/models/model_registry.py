"""Qwen portfolio model registry — local path resolution via SCENARIO_MODELS_ROOT."""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

import yaml

DEFAULT_PORTFOLIO_PATH = Path(__file__).resolve().parents[2] / "configs" / "models" / "qwen_portfolio.yaml"
MODELS_ROOT_ENV = "SCENARIO_MODELS_ROOT"
TRAINED_ROOT_ENV = "SCENARIO_TRAINED_MODELS_ROOT"


@dataclass
class ModelEntry:
    model_id: str
    hub_id: str
    tier: str

    def local_dir(self, models_root: Path) -> Path:
        return models_root / self.hub_id


class ModelRegistry:
    def __init__(self, raw: Dict[str, Any], portfolio_path: Optional[Path] = None) -> None:
        self._raw = raw
        self.portfolio_path = portfolio_path or DEFAULT_PORTFOLIO_PATH
        self.models_root_env = raw.get("models_root_env", MODELS_ROOT_ENV)
        self.trained_root_env = raw.get("trained_root_env", TRAINED_ROOT_ENV)
        self._entries: Dict[str, ModelEntry] = {}
        for model_id, spec in raw.get("models", {}).items():
            self._entries[model_id] = ModelEntry(
                model_id=model_id,
                hub_id=spec["hub_id"],
                tier=spec.get("tier", "mid"),
            )
        self._hub_to_id = {e.hub_id: e.model_id for e in self._entries.values()}

    @classmethod
    def load(cls, path: Optional[Path] = None) -> "ModelRegistry":
        portfolio = path or DEFAULT_PORTFOLIO_PATH
        with open(portfolio, encoding="utf-8") as f:
            raw = yaml.safe_load(f)
        return cls(raw, portfolio_path=portfolio)

    def get(self, model_id: str) -> ModelEntry:
        if model_id not in self._entries:
            raise KeyError(f"Unknown model_id: {model_id!r}. Known: {list(self._entries)}")
        return self._entries[model_id]

    def get_by_tier(self, tier: str) -> List[ModelEntry]:
        tier_ids = self._raw.get("tiers", {}).get(tier, [])
        return [self.get(mid) for mid in tier_ids]

    def all_entries(self) -> List[ModelEntry]:
        return list(self._entries.values())

    @staticmethod
    def require_models_root(env_name: str = MODELS_ROOT_ENV) -> Path:
        value = os.environ.get(env_name)
        if not value:
            raise EnvironmentError(
                f"{env_name} is not set. Export it to the directory containing "
                f"pre-downloaded models (e.g. .../Qwen/Qwen3-0.6B). See docs/models.md."
            )
        root = Path(value)
        if not root.is_dir():
            raise FileNotFoundError(f"{env_name}={value} is not a directory")
        return root

    @staticmethod
    def trained_root(env_name: str = TRAINED_ROOT_ENV) -> Optional[Path]:
        value = os.environ.get(env_name)
        if not value:
            return None
        return Path(value)

    def resolve_path(self, model_id_or_hub: str) -> Path:
        """Resolve registry key or hub id to local directory under SCENARIO_MODELS_ROOT."""
        root = self.require_models_root(self.models_root_env)
        if model_id_or_hub in self._entries:
            entry = self.get(model_id_or_hub)
            return entry.local_dir(root)
        if model_id_or_hub in self._hub_to_id:
            return self.get(self._hub_to_id[model_id_or_hub]).local_dir(root)
        # Legacy: treat as path or hub slug under root
        candidate = root / model_id_or_hub
        if candidate.is_dir():
            return candidate
        return root / model_id_or_hub

    def resolve_model_name_or_path(self, config: Dict[str, Any]) -> str:
        """Inject resolved local path into config for model loading."""
        if config.get("model_id"):
            local = self.resolve_path(config["model_id"])
            if not local.is_dir():
                raise FileNotFoundError(
                    f"Model {config['model_id']!r} not found at {local}. "
                    f"Run: python scripts/download_models.py --model-id {config['model_id']}"
                )
            return str(local)
        if config.get("model_name_or_path"):
            path = Path(config["model_name_or_path"])
            if path.is_dir():
                return str(path)
            # Hub id — try local cache first
            try:
                local = self.resolve_path(config["model_name_or_path"])
                if local.is_dir():
                    return str(local)
            except (EnvironmentError, KeyError):
                pass
            return config["model_name_or_path"]
        raise ValueError("Config requires model_id or model_name_or_path")

    def to_train_config(self, model_id: str, base_config: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        cfg = dict(base_config or {})
        cfg["model_id"] = model_id
        cfg["model_name_or_path"] = self.resolve_model_name_or_path({"model_id": model_id})
        entry = self.get(model_id)
        cfg.setdefault("model_tier", entry.tier)
        cfg.setdefault("hub_id", entry.hub_id)
        return cfg
