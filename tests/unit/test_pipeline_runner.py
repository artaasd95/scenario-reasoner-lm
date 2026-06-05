"""Pipeline runner unit tests on synthetic config (S9-DP-03)."""

from __future__ import annotations

from src.data.pipeline.config import PipelineConfig
from src.data.pipeline.runner import PipelineRunner
from src.data.sources.base import DataSource


class _SyntheticSource(DataSource):
    def iter_rows(self):
        yield {
            "path_id": "syn-1",
            "theta": {
                "chain_length": 2,
                "intervention_type": "direct",
                "domain": "physical",
                "difficulty": "easy",
            },
            "theta_kind": "causal",
            "primary_quality_score": 0.8,
        }
        yield {
            "path_id": "syn-2",
            "theta": {},
            "infeasible": True,
            "primary_quality_score": 0.1,
        }


class TestPipelineRunner:
    def test_synthetic_train_pipeline(self, tmp_path):
        cfg = PipelineConfig(
            mode="train",
            stages=["normalize", "measure", "label", "filter"],
            output_dir=str(tmp_path),
            filter={"drop_infeasible": True, "feasibility_threshold": 0.2},
        )
        result = PipelineRunner(cfg, source=_SyntheticSource()).run()
        assert result["stats"]["normalized"] >= 1
        assert result["stats"]["dropped"] >= 1

    def test_eval_writes_measurement(self, tmp_path):
        cfg = PipelineConfig(
            mode="eval",
            stages=["normalize", "measure"],
            output_dir=str(tmp_path),
        )
        PipelineRunner(cfg, source=_SyntheticSource()).run()
        assert (tmp_path / "scenario_measurement.json").is_file()
