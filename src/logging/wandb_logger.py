"""
Weights & Biases logger for Scenario Reasoner LM.

Provides a thin, safe wrapper around the ``wandb`` library.  If ``wandb``
is not installed the logger silently operates in no-op mode so that the
rest of the codebase does not need to handle optional imports.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, Optional

logger = logging.getLogger(__name__)

try:
    import wandb as _wandb
    _WANDB_AVAILABLE = True
except ImportError:
    _wandb = None  # type: ignore[assignment]
    _WANDB_AVAILABLE = False


class WandbLogger:
    """
    Experiment logger backed by Weights & Biases.

    When ``wandb`` is not installed, all methods become silent no-ops,
    allowing training code to be written without import guards.

    Example::

        logger = WandbLogger(project="scenario-reasoner", name="run-01")
        logger.log_config({"lr": 1e-4, "epochs": 10})

        for step, batch in enumerate(train_loader):
            ...
            logger.log_step(step, {"loss": loss.item()})

        logger.finish()

    Args:
        project: W&B project name.
        name: Run name shown in the W&B UI.
        config: Optional dictionary of hyper-parameters to attach to the run.
        tags: Optional list of run tags.
        entity: W&B entity (team or user name).
        mode: W&B run mode — ``"online"`` | ``"offline"`` | ``"disabled"``.
    """

    def __init__(
        self,
        project: str = "scenario-reasoner-lm",
        name: Optional[str] = None,
        config: Optional[Dict[str, Any]] = None,
        tags: Optional[list] = None,
        entity: Optional[str] = None,
        mode: str = "online",
    ) -> None:
        self._active = False

        if not _WANDB_AVAILABLE:
            logger.warning(
                "wandb is not installed; WandbLogger is running in no-op mode. "
                "Install with: pip install wandb"
            )
            return

        try:
            _wandb.init(
                project=project,
                name=name,
                config=config or {},
                tags=tags,
                entity=entity,
                mode=mode,
            )
            self._active = True
        except Exception as exc:
            logger.error("Failed to initialise wandb run: %s", exc)

    @property
    def is_active(self) -> bool:
        """``True`` if wandb was successfully initialised."""
        return self._active

    def log_config(self, config: Dict[str, Any]) -> None:
        """
        Update the run configuration with additional key-value pairs.

        Args:
            config: Dictionary of configuration parameters.
        """
        if self._active:
            _wandb.config.update(config, allow_val_change=True)

    def log_step(
        self,
        step: int,
        metrics: Dict[str, Any],
        prefix: str = "train",
    ) -> None:
        """
        Log per-step metrics to W&B.

        Args:
            step: Global training step.
            metrics: Metric name → value dictionary.
            prefix: Prefix prepended to all metric names (e.g. ``"train/loss"``).
        """
        if self._active:
            prefixed = {f"{prefix}/{k}": v for k, v in metrics.items()}
            _wandb.log(prefixed, step=step)

    def log_epoch(
        self,
        epoch: int,
        metrics: Dict[str, Any],
        prefix: str = "epoch",
    ) -> None:
        """
        Log per-epoch summary metrics.

        Args:
            epoch: Epoch index.
            metrics: Metric name → value dictionary.
            prefix: Prefix prepended to all metric names.
        """
        if self._active:
            prefixed = {f"{prefix}/{k}": v for k, v in metrics.items()}
            prefixed["epoch"] = epoch
            _wandb.log(prefixed)

    def log_monitoring(
        self,
        monitor_name: str,
        stats: Dict[str, Any],
        step: Optional[int] = None,
    ) -> None:
        """
        Log monitoring statistics from a CoT/ToT/Aha monitor.

        Args:
            monitor_name: Monitor identifier (e.g. ``"cot"``, ``"tot"``, ``"aha"``).
            stats: Statistics dictionary from ``monitor.get_stats()``.
            step: Optional training step.
        """
        if self._active:
            prefixed = {f"monitor/{monitor_name}/{k}": v for k, v in stats.items()
                        if isinstance(v, (int, float))}
            if step is not None:
                _wandb.log(prefixed, step=step)
            else:
                _wandb.log(prefixed)

    def log_artifact(
        self,
        path: str,
        name: str,
        artifact_type: str = "dataset",
    ) -> None:
        """
        Upload a file or directory as a W&B artifact.

        Args:
            path: Local path to the file or directory.
            name: Artifact name in W&B.
            artifact_type: Artifact type label (e.g. ``"dataset"``).
        """
        if self._active:
            artifact = _wandb.Artifact(name=name, type=artifact_type)
            artifact.add_file(path)
            _wandb.log_artifact(artifact)

    def finish(self) -> None:
        """Mark the W&B run as completed and sync remaining data."""
        if self._active:
            _wandb.finish()
            self._active = False

    def __enter__(self) -> "WandbLogger":
        return self

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        self.finish()
