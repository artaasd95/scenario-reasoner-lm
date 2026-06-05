"""BaseExperimentLogger — pluggable experiment backends (S10-04)."""

from __future__ import annotations

import json
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Dict, Optional

from src.logging.local_logger import LocalLogger


class BaseExperimentLogger(ABC):
    @abstractmethod
    def log_config(self, config: Dict[str, Any]) -> None:
        ...

    @abstractmethod
    def log_metrics(
        self,
        metrics: Dict[str, Any],
        *,
        step: Optional[int] = None,
        epoch: Optional[int] = None,
        prefix: str = "train",
    ) -> None:
        ...

    @abstractmethod
    def close(self) -> None:
        ...


class LocalExperimentLogger(BaseExperimentLogger):
    def __init__(self, name: str, log_dir: str) -> None:
        self._logger = LocalLogger(name=name, log_dir=log_dir)

    def log_config(self, config: Dict[str, Any]) -> None:
        self._logger.log_config(config)

    def log_metrics(
        self,
        metrics: Dict[str, Any],
        *,
        step: Optional[int] = None,
        epoch: Optional[int] = None,
        prefix: str = "train",
    ) -> None:
        self._logger.log_metrics(metrics, step=step, epoch=epoch, prefix=prefix)

    def close(self) -> None:
        self._logger.close()


class WandbExperimentLogger(BaseExperimentLogger):
    def __init__(self, project: str, name: Optional[str] = None, config: Optional[Dict] = None) -> None:
        from src.logging.wandb_logger import WandbLogger

        self._logger = WandbLogger(project=project, name=name, config=config or {})

        self._available = True

    def log_config(self, config: Dict[str, Any]) -> None:
        pass  # config set at init

    def log_metrics(
        self,
        metrics: Dict[str, Any],
        *,
        step: Optional[int] = None,
        epoch: Optional[int] = None,
        prefix: str = "train",
    ) -> None:
        self._logger.log_step(step or 0, metrics, prefix=prefix)

    def close(self) -> None:
        self._logger.finish()


class CometExperimentLogger(BaseExperimentLogger):
    """Stub — records to local JSONL when comet_ml is unavailable."""

    def __init__(self, log_dir: str) -> None:
        self._path = Path(log_dir) / "comet_stub.jsonl"
        self._path.parent.mkdir(parents=True, exist_ok=True)
        self._fh = self._path.open("a", encoding="utf-8")

    def log_config(self, config: Dict[str, Any]) -> None:
        self._write({"event": "config", "data": config})

    def log_metrics(
        self,
        metrics: Dict[str, Any],
        *,
        step: Optional[int] = None,
        epoch: Optional[int] = None,
        prefix: str = "train",
    ) -> None:
        self._write({"event": "metrics", "step": step, "epoch": epoch, "prefix": prefix, "metrics": metrics})

    def _write(self, record: Dict[str, Any]) -> None:
        self._fh.write(json.dumps(record) + "\n")
        self._fh.flush()

    def close(self) -> None:
        self._fh.close()


class SQLExperimentLogger(BaseExperimentLogger):
    """Append-only JSONL shipper suitable for SQL ingest."""

    def __init__(self, log_dir: str, table: str = "experiment_metrics") -> None:
        self._path = Path(log_dir) / f"{table}.jsonl"
        self._path.parent.mkdir(parents=True, exist_ok=True)
        self._fh = self._path.open("a", encoding="utf-8")

    def log_config(self, config: Dict[str, Any]) -> None:
        self._fh.write(json.dumps({"event": "config", "data": config}) + "\n")

    def log_metrics(
        self,
        metrics: Dict[str, Any],
        *,
        step: Optional[int] = None,
        epoch: Optional[int] = None,
        prefix: str = "train",
    ) -> None:
        self._fh.write(
            json.dumps({"event": "metrics", "step": step, "epoch": epoch, "prefix": prefix, "metrics": metrics})
            + "\n"
        )
        self._fh.flush()

    def close(self) -> None:
        self._fh.close()


class ExperimentLoggerFactory:
    @staticmethod
    def create(backend: str = "local", **kwargs: Any) -> BaseExperimentLogger:
        if backend == "local":
            return LocalExperimentLogger(kwargs.get("name", "exp"), kwargs.get("log_dir", "./logs"))
        if backend == "wandb":
            return WandbExperimentLogger(
                project=kwargs.get("project", "scenario-reasoner-lm"),
                name=kwargs.get("name"),
                config=kwargs.get("config"),
            )
        if backend == "comet":
            return CometExperimentLogger(kwargs.get("log_dir", "./logs"))
        if backend == "sql":
            return SQLExperimentLogger(kwargs.get("log_dir", "./logs"), kwargs.get("table", "experiment_metrics"))
        raise ValueError(f"Unknown backend: {backend}")
