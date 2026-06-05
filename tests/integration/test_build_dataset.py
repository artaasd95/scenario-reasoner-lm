"""Integration test for build_dataset CLI (S9-DP-05)."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

_REPO = Path(__file__).resolve().parents[2]


class TestBuildDataset:
    def test_train_config_smoke(self, tmp_path):
        out = tmp_path / "processed"
        config = _REPO / "configs" / "data" / "train.yaml"
        # Run pipeline via subprocess-free import for speed
        from src.data.pipeline.config import load_pipeline_config
        from src.data.pipeline.runner import PipelineRunner
        from src.data.pipeline.source_factory import build_source

        cfg = load_pipeline_config(config)
        cfg.output_dir = str(out)
        result = PipelineRunner(cfg, source=build_source(cfg.source)).run()
        assert result["stats"]["loaded"] > 0

    def test_eval_config_writes_measurement(self, tmp_path):
        from src.data.pipeline.config import load_pipeline_config
        from src.data.pipeline.runner import PipelineRunner
        from src.data.pipeline.source_factory import build_source

        cfg = load_pipeline_config(_REPO / "configs" / "data" / "eval.yaml")
        cfg.output_dir = str(tmp_path / "eval")
        PipelineRunner(cfg, source=build_source(cfg.source)).run()
        assert (tmp_path / "eval" / "scenario_measurement.json").is_file()
