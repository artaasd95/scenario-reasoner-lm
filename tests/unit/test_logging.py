"""
Unit tests for local structured logging.
"""

from __future__ import annotations

import json

from src.logging.local_logger import LocalLogger


def _read_metrics_records(log_dir):
    metrics_files = list(log_dir.glob("*_metrics.jsonl"))
    assert len(metrics_files) == 1
    return [
        json.loads(line)
        for line in metrics_files[0].read_text(encoding="utf-8").splitlines()
    ]


class TestLocalLogger:
    def test_writes_structured_records(self, tmp_path):
        logger = LocalLogger(name="unit", log_dir=str(tmp_path), use_console=False)

        logger.log_config({"learning_rate": 1e-4})
        logger.log_step(step=3, metrics={"loss": 1.25}, prefix="train")
        logger.log_epoch(epoch=0, metrics={"val_loss": 0.75}, prefix="eval")
        logger.log_monitoring("cot", {"total_samples": 2}, step=3)
        logger.close()

        records = _read_metrics_records(tmp_path)
        assert [record["event"] for record in records] == [
            "config",
            "step",
            "epoch",
            "monitoring",
        ]
        assert records[1]["step"] == 3
        assert records[1]["prefix"] == "train"
        assert records[1]["metrics"]["loss"] == 1.25
        assert records[3]["monitor"] == "cot"
        assert all("timestamp" in record for record in records)

    def test_context_manager_closes_and_close_is_idempotent(self, tmp_path):
        with LocalLogger(name="ctx", log_dir=str(tmp_path), use_console=False) as logger:
            logger.log_step(step=0, metrics={"count": 1}, prefix="train")

        logger.close()
        logger.close()

        records = _read_metrics_records(tmp_path)
        assert records[0]["event"] == "step"
        assert records[0]["metrics"] == {"count": 1}

    def test_message_methods_write_log_file(self, tmp_path):
        logger = LocalLogger(name="messages", log_dir=str(tmp_path), use_console=False)

        logger.info("hello %s", "world")
        logger.warning("careful")
        logger.error("boom")
        logger.close()

        log_files = [path for path in tmp_path.glob("*.log") if "metrics" not in path.name]
        assert len(log_files) == 1
        log_text = log_files[0].read_text(encoding="utf-8")
        assert "hello world" in log_text
        assert "careful" in log_text
        assert "boom" in log_text
