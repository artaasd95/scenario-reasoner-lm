"""Training backend factory — TRL (default) or Unsloth."""

from __future__ import annotations

from typing import Dict

from src.training.backends.base import TrainerBackend
from src.training.backends.trl_backend import TrlBackend
from src.training.backends.unsloth_trainer import UnslothBackend

_BACKENDS: Dict[str, TrainerBackend] = {
    "trl": TrlBackend(),
    "unsloth": UnslothBackend(),
}


def get_trainer_backend(name: str) -> TrainerBackend:
    key = (name or "trl").lower()
    if key not in _BACKENDS:
        raise ValueError(
            f"Unknown training backend: {name!r}. Choose from: {list(_BACKENDS)}"
        )
    return _BACKENDS[key]


__all__ = ["TrainerBackend", "get_trainer_backend", "TrlBackend", "UnslothBackend"]
