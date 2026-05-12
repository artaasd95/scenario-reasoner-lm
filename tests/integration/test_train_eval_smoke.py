"""
Smoke tests for train/eval CLI entrypoints.

Heavy model loading, generation, and DPO/evaluation execution are patched so
these tests only verify script wiring, config handling, logging, and outputs.
"""

from __future__ import annotations

import json
import importlib.util
import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace


def _load_script(name: str):
    script_path = Path(__file__).resolve().parents[2] / "scripts" / f"{name}.py"
    spec = importlib.util.spec_from_file_location(f"_smoke_{name}", script_path)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _write_config(path: Path, output_dir: Path) -> None:
    path.write_text(
        json.dumps({
            "experiment_name": "smoke",
            "project": "scenario-reasoner-lm",
            "model_name_or_path": "dummy-model",
            "use_qlora": False,
            "scenario": {
                "chain_lengths": [2],
                "intervention_types": ["direct"],
                "domains": ["physical"],
                "difficulties": ["easy"],
                "n_per_combo": 1,
            },
            "preference_builder": {
                "num_samples": 2,
                "temperature": 0.1,
                "max_new_tokens": 8,
            },
            "reward_weights": {
                "alpha_cot": 0.15,
                "beta_tot": 0.10,
                "gamma_aha": 0.05,
            },
            "data": {
                "train_frac": 1.0,
                "seed": 7,
            },
            "output_dir": str(output_dir),
        }),
        encoding="utf-8",
    )


def _read_metrics_records(log_dir: Path):
    metrics_files = list(log_dir.glob("*_metrics.jsonl"))
    assert len(metrics_files) == 1
    return [
        json.loads(line)
        for line in metrics_files[0].read_text(encoding="utf-8").splitlines()
    ]


def _install_lightweight_torch(monkeypatch) -> None:
    torch_module = ModuleType("torch")

    class _Tensor:
        pass

    class _Dataset:
        pass

    class _DataLoader:
        def __init__(self, dataset, *args, **kwargs):
            self.dataset = dataset

        def __iter__(self):
            return iter(self.dataset)

    torch_module.Tensor = _Tensor
    torch_module.stack = lambda values: list(values)
    torch_module.bfloat16 = "bfloat16"

    utils_module = ModuleType("torch.utils")
    data_module = ModuleType("torch.utils.data")
    data_module.Dataset = _Dataset
    data_module.DataLoader = _DataLoader
    utils_module.data = data_module
    torch_module.utils = utils_module

    monkeypatch.setitem(sys.modules, "torch", torch_module)
    monkeypatch.setitem(sys.modules, "torch.utils", utils_module)
    monkeypatch.setitem(sys.modules, "torch.utils.data", data_module)


class _DummyWrapper:
    def load(self):
        return SimpleNamespace(device="cpu"), SimpleNamespace()


def test_train_main_smoke(monkeypatch, tmp_path):
    from src.models.model_wrapper import ModelWrapper
    from src.training.preference_builder import PreferenceBuilder
    from src.training.rlhf_trainer import RLHFTrainer

    train_script = _load_script("train")
    output_dir = tmp_path / "train-run"
    config_path = tmp_path / "config.json"
    _write_config(config_path, output_dir)

    _install_lightweight_torch(monkeypatch)
    monkeypatch.setattr(
        sys,
        "argv",
        ["train.py", "--config", str(config_path), "--output-dir", str(output_dir)],
    )
    monkeypatch.setattr(
        ModelWrapper,
        "from_config",
        classmethod(lambda cls, config: _DummyWrapper()),
    )
    monkeypatch.setattr(
        PreferenceBuilder,
        "build_dataset",
        lambda self, prompts, expected_answers=None, thetas=None: [{
            "prompt": prompts[0],
            "chosen": "Step 1: A causes B. Therefore B.",
            "rejected": "unknown",
        }],
    )
    monkeypatch.setattr(
        RLHFTrainer,
        "train",
        lambda self: str(self.output_dir / "dpo_checkpoint"),
    )

    train_script.main()

    records = _read_metrics_records(output_dir / "logs")
    step_records = [record for record in records if record["event"] == "step"]
    assert step_records[0]["metrics"] == {"preference_pairs": 1}
    assert step_records[1]["metrics"]["checkpoint"].endswith("dpo_checkpoint")


class _DummyModel:
    def eval(self):
        return None


class _DummyTokenizer:
    pad_token = None
    eos_token = "<eos>"


class _DummyAutoTokenizer:
    @staticmethod
    def from_pretrained(*args, **kwargs):
        return _DummyTokenizer()


class _DummyAutoModelForCausalLM:
    @staticmethod
    def from_pretrained(*args, **kwargs):
        return SimpleNamespace()


class _DummyPeftModel:
    @staticmethod
    def from_pretrained(*args, **kwargs):
        return _DummyModel()


def test_evaluate_main_smoke(monkeypatch, tmp_path):
    from src.evaluation.robustness_eval import RobustnessEvaluator

    evaluate_script = _load_script("evaluate")
    output_dir = tmp_path / "eval-run"
    config_path = tmp_path / "config.json"
    _write_config(config_path, tmp_path / "train-run")

    torch_module = ModuleType("torch")
    torch_module.bfloat16 = "bfloat16"
    peft_module = ModuleType("peft")
    peft_module.PeftModel = _DummyPeftModel
    transformers_module = ModuleType("transformers")
    transformers_module.AutoModelForCausalLM = _DummyAutoModelForCausalLM
    transformers_module.AutoTokenizer = _DummyAutoTokenizer

    monkeypatch.setitem(sys.modules, "torch", torch_module)
    monkeypatch.setitem(sys.modules, "peft", peft_module)
    monkeypatch.setitem(sys.modules, "transformers", transformers_module)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "evaluate.py",
            "--config",
            str(config_path),
            "--checkpoint",
            str(tmp_path / "checkpoint"),
            "--output-dir",
            str(output_dir),
            "--n-eval",
            "1",
        ],
    )
    monkeypatch.setattr(
        RobustnessEvaluator,
        "evaluate",
        lambda self: {
            "per_theta": [],
            "aggregate": {
                "causal_chain_accuracy": 1.0,
                "counterfactual_validity": 1.0,
                "trajectory_consistency": 1.0,
            },
            "theta_count": len(self.theta_grid),
            "n_eval": self.n_eval,
        },
    )

    def _save_report(self, report, output_path):
        Path(output_path).write_text(json.dumps(report), encoding="utf-8")

    monkeypatch.setattr(RobustnessEvaluator, "save_report", _save_report)

    evaluate_script.main()

    report = json.loads((output_dir / "robustness_report.json").read_text(encoding="utf-8"))
    records = _read_metrics_records(output_dir / "logs")
    eval_step = next(record for record in records if record["event"] == "step")

    assert report["aggregate"]["causal_chain_accuracy"] == 1.0
    assert eval_step["prefix"] == "eval"
    assert eval_step["metrics"]["trajectory_consistency"] == 1.0
