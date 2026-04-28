"""
Local file-based logger for Scenario Reasoner LM.

Provides structured logging to console and JSON-Lines files, with helpers
for logging training steps, epoch summaries, configurations, and
monitoring statistics.
"""

from __future__ import annotations

import json
import logging
import os
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Optional


class LocalLogger:
    """
    Structured local logger supporting both console output and JSONL file logging.

    Creates one log file per run (``<log_dir>/<name>_<timestamp>.log``) and a
    corresponding metrics file (``<log_dir>/<name>_<timestamp>_metrics.jsonl``).

    Example::

        logger = LocalLogger(name="exp01", log_dir="./logs")
        logger.log_config({"lr": 1e-4, "batch_size": 16})

        for step, batch in enumerate(dataloader):
            ...
            logger.log_step(step, {"loss": loss.item()})

        logger.log_epoch(0, {"val_loss": 0.8, "exact_match": 0.65})

    Args:
        name: Experiment name used as prefix for log file names.
        log_dir: Directory in which to write log files.  Created if absent.
        level: Python logging level (e.g. ``logging.INFO``).
        use_console: Whether to also log to stdout.
    """

    def __init__(
        self,
        name: str = "experiment",
        log_dir: str = "./logs",
        level: int = logging.INFO,
        use_console: bool = True,
    ) -> None:
        self.name = name
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)

        timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        log_file = self.log_dir / f"{name}_{timestamp}.log"
        self._metrics_file = self.log_dir / f"{name}_{timestamp}_metrics.jsonl"

        self._python_logger = logging.getLogger(f"scenario_reasoner.{name}.{timestamp}")
        self._python_logger.setLevel(level)
        self._python_logger.propagate = False

        formatter = logging.Formatter(
            fmt="%(asctime)s [%(levelname)s] %(message)s",
            datefmt="%Y-%m-%dT%H:%M:%SZ",
        )

        file_handler = logging.FileHandler(str(log_file))
        file_handler.setFormatter(formatter)
        self._python_logger.addHandler(file_handler)

        if use_console:
            console_handler = logging.StreamHandler(sys.stdout)
            console_handler.setFormatter(formatter)
            self._python_logger.addHandler(console_handler)

        self._metrics_fh = open(str(self._metrics_file), "a", encoding="utf-8")

    def log_config(self, config: Dict[str, Any]) -> None:
        """
        Log an experiment configuration dictionary.

        Args:
            config: Arbitrary key-value configuration pairs.
        """
        self._python_logger.info("CONFIG: %s", json.dumps(config))
        self._write_record({"event": "config", "data": config})

    def log_step(
        self,
        step: int,
        metrics: Dict[str, Any],
        prefix: str = "train",
    ) -> None:
        """
        Log metrics for a single training step.

        Args:
            step: Global training step index.
            metrics: Dictionary of metric name → value.
            prefix: Log category prefix (e.g. ``"train"``, ``"eval"``).
        """
        msg = " | ".join(f"{k}={v:.4f}" if isinstance(v, float) else f"{k}={v}"
                         for k, v in metrics.items())
        self._python_logger.info("[%s] step=%d | %s", prefix, step, msg)
        self._write_record({"event": "step", "step": step, "prefix": prefix, "metrics": metrics})

    def log_epoch(
        self,
        epoch: int,
        metrics: Dict[str, Any],
        prefix: str = "epoch",
    ) -> None:
        """
        Log summary metrics for a completed epoch.

        Args:
            epoch: Epoch index.
            metrics: Dictionary of metric name → value.
            prefix: Log category prefix.
        """
        msg = " | ".join(f"{k}={v:.4f}" if isinstance(v, float) else f"{k}={v}"
                         for k, v in metrics.items())
        self._python_logger.info("[%s] epoch=%d | %s", prefix, epoch, msg)
        self._write_record({"event": "epoch", "epoch": epoch, "prefix": prefix, "metrics": metrics})

    def log_monitoring(
        self,
        monitor_name: str,
        stats: Dict[str, Any],
        step: Optional[int] = None,
    ) -> None:
        """
        Log monitoring statistics from a CoT/ToT/Aha monitor.

        Args:
            monitor_name: Name of the monitor (e.g. ``"cot"``, ``"tot"``, ``"aha"``).
            stats: Statistics dictionary returned by ``monitor.get_stats()``.
            step: Optional training step index.
        """
        self._python_logger.info(
            "[monitor:%s] step=%s | %s", monitor_name, step, json.dumps(stats)
        )
        self._write_record({
            "event": "monitoring",
            "monitor": monitor_name,
            "step": step,
            "stats": stats,
        })

    def info(self, msg: str, *args: Any, **kwargs: Any) -> None:
        """Forward an info-level message to the Python logger."""
        self._python_logger.info(msg, *args, **kwargs)

    def warning(self, msg: str, *args: Any, **kwargs: Any) -> None:
        """Forward a warning-level message to the Python logger."""
        self._python_logger.warning(msg, *args, **kwargs)

    def error(self, msg: str, *args: Any, **kwargs: Any) -> None:
        """Forward an error-level message to the Python logger."""
        self._python_logger.error(msg, *args, **kwargs)

    def close(self) -> None:
        """Flush and close all log file handles."""
        self._metrics_fh.flush()
        self._metrics_fh.close()
        for handler in list(self._python_logger.handlers):
            handler.close()
            self._python_logger.removeHandler(handler)

    def _write_record(self, record: Dict[str, Any]) -> None:
        """Write a JSONL record to the metrics file."""
        record["timestamp"] = datetime.now(timezone.utc).isoformat()
        self._metrics_fh.write(json.dumps(record) + "\n")
        self._metrics_fh.flush()

    def __enter__(self) -> "LocalLogger":
        return self

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        self.close()
