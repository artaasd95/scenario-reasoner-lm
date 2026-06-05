"""BaseExperimentLogger tests (S10-04)."""

from __future__ import annotations

from src.logging.experiment_logger import (
    CometExperimentLogger,
    ExperimentLoggerFactory,
    SQLExperimentLogger,
)


class TestExperimentLogger:
    def test_local_factory(self, tmp_path):
        logger = ExperimentLoggerFactory.create("local", name="t", log_dir=str(tmp_path))
        logger.log_metrics({"loss": 1.0}, step=0)
        logger.close()

    def test_comet_stub(self, tmp_path):
        logger = CometExperimentLogger(str(tmp_path))
        logger.log_metrics({"acc": 0.5}, step=1)
        logger.close()
        assert (tmp_path / "comet_stub.jsonl").is_file()

    def test_sql_shipper(self, tmp_path):
        logger = SQLExperimentLogger(str(tmp_path))
        logger.log_config({"lr": 1e-4})
        logger.close()
