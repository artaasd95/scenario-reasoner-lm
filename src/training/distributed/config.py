"""Distributed training config — DDP/FSDP; Ray optional (S10-06)."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Optional


@dataclass
class DistributedTrainingConfig:
    """
    Default single-process; enable DDP/FSDP explicitly.

    CI stays single-device unless overridden in integration tests.
    """

    enabled: bool = False
    backend: str = "ddp"  # ddp | fsdp | ray
    world_size: int = 1
    local_rank: int = 0
    ray_address: Optional[str] = None
    fsdp_config: Dict[str, Any] = field(default_factory=dict)

    @classmethod
    def single_process(cls) -> "DistributedTrainingConfig":
        return cls(enabled=False, world_size=1)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "enabled": self.enabled,
            "backend": self.backend,
            "world_size": self.world_size,
            "local_rank": self.local_rank,
            "ray_address": self.ray_address,
            "fsdp_config": dict(self.fsdp_config),
        }

    def validate_ci_safe(self) -> None:
        if self.enabled and self.world_size > 1:
            raise ValueError("CI must use single-process distributed config")
