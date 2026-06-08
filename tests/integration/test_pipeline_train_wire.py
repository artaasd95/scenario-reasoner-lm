"""Integration: build_dataset → train mini-run with pipeline JSONL (SR-03)."""

from __future__ import annotations

import json
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest


@pytest.mark.smoke
def test_build_dataset_then_train_pipeline_wire(monkeypatch, tmp_path):
    from src.data.pipeline.config import load_pipeline_config
    from src.data.pipeline.runner import PipelineRunner
    from src.data.pipeline.source_factory import build_source
    from src.training.backends.trl_backend import TrlBackend
    from src.training.preference_builder import PreferenceBuilder
    from tests.integration.test_smoke_train_eval import _load_script, _write_config

    cfg = load_pipeline_config("configs/data/train.yaml")
    out_dir = tmp_path / "pipeline_out"
    cfg.output_dir = str(out_dir)
    source = build_source({"type": "generator", "include_game": True})
    result = PipelineRunner(cfg, source=source).run()
    jsonl_path = Path(result["output"])
    assert jsonl_path.is_file()

    train_script = _load_script("train")
    run_dir = tmp_path / "train_run"
    config_path = tmp_path / "train_config.json"
    _write_config(config_path, run_dir)
    config = json.loads(config_path.read_text(encoding="utf-8"))
    config.setdefault("training", {})["data_source"] = "pipeline"
    config["pipeline_train_jsonl"] = str(jsonl_path)
    config_path.write_text(json.dumps(config), encoding="utf-8")

    monkeypatch.setattr(
        sys,
        "argv",
        [
            "train.py",
            "--config",
            str(config_path),
            "--output-dir",
            str(run_dir),
            "--data-config",
            "configs/data/train.yaml",
        ],
    )
    monkeypatch.setattr(
        TrlBackend,
        "load_model",
        lambda self, cfg: (SimpleNamespace(device="cpu"), SimpleNamespace()),
    )
    monkeypatch.setattr(
        PreferenceBuilder,
        "build_dataset",
        lambda self, prompts, expected_answers=None, thetas=None: [{
            "prompt": prompts[0],
            "chosen": "ok",
            "rejected": "bad",
        }],
    )
    monkeypatch.setattr(
        TrlBackend,
        "train",
        lambda self, model, tokenizer, preference_data, cfg, **kwargs: str(run_dir / "dpo_checkpoint"),
    )

    from src.data.pipeline.runner import PipelineRunner as PR

    def _fake_run(self, source=None):
        return {"output": str(jsonl_path), "stats": {"loaded": 4}}

    monkeypatch.setattr(PR, "run", _fake_run)

    train_script.main()
    assert (run_dir / "logs").is_dir()
