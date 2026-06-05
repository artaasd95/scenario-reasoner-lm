"""Integration: generator source → pipeline → artifact (S9-DP-04)."""

from __future__ import annotations

from src.data.pipeline.config import load_pipeline_config
from src.data.pipeline.runner import PipelineRunner
from src.data.pipeline.source_factory import build_source


class TestGeneratorSourcePipeline:
    def test_generator_to_pipeline(self, tmp_path):
        cfg = load_pipeline_config("configs/data/train.yaml")
        cfg.output_dir = str(tmp_path / "out")
        source = build_source({"type": "generator", "include_game": True})
        result = PipelineRunner(cfg, source=source).run()
        assert result["stats"]["loaded"] >= 4
        assert (tmp_path / "out" / "pipeline_train.jsonl").is_file()
