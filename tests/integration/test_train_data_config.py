"""Smoke: train pipeline data-config path (S9-DP-06)."""

from __future__ import annotations

from src.data.pipeline.config import load_pipeline_config
from src.data.pipeline.runner import PipelineRunner
from src.data.pipeline.source_factory import build_source


class TestTrainDataConfig:
    def test_bundled_game_financial_fixtures_via_generator(self, tmp_path):
        cfg = load_pipeline_config("configs/data/train.yaml")
        cfg.output_dir = str(tmp_path)
        source = build_source(cfg.source)
        result = PipelineRunner(cfg, source=source).run()
        assert result["stats"]["loaded"] >= 5
        assert result["stats"]["measured"] >= 1
