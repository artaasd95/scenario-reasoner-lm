"""
Smoke tests for Unsloth training backend dispatch.

Mocks FastLanguageModel and TRL so no GPU or unsloth install is required.
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


def _write_yaml_config(path: Path, output_dir: Path) -> None:
    import yaml

    path.write_text(
        yaml.dump({
            "experiment_name": "unsloth_smoke",
            "model_name_or_path": "dummy-local-model",
            "training": {
                "backend": "unsloth",
                "policy": "causal_default",
                "data_source": "distilled",
                "distilled_manifest": str(
                    Path(__file__).resolve().parents[1]
                    / "fixtures"
                    / "distilled"
                    / "scenario_traces_v1"
                    / "manifest.json"
                ),
            },
            "use_qlora": True,
            "output_dir": str(output_dir),
            "data": {"seed": 7},
        }),
        encoding="utf-8",
    )


def test_unsloth_backend_train_smoke(monkeypatch, tmp_path):
    from src.training.backends.unsloth_trainer import UnslothBackend
    from src.models.loaders.unified import UnifiedModelLoader

    train_script = _load_script("train")
    output_dir = tmp_path / "unsloth-run"
    config_path = tmp_path / "config.yaml"
    _write_yaml_config(config_path, output_dir)

    checkpoint_dir = output_dir / "dpo_checkpoint"
    checkpoint_dir.mkdir(parents=True)

    class _FakeModel:
        def save_pretrained(self, path):
            Path(path).mkdir(parents=True, exist_ok=True)
            (Path(path) / "adapter_config.json").write_text(
                json.dumps({
                    "peft_type": "LORA",
                    "base_model_name_or_path": "dummy-local-model",
                }),
                encoding="utf-8",
            )

    class _FakeTokenizer:
        pad_token = None
        eos_token = "<eos>"

        def save_pretrained(self, path):
            Path(path).mkdir(parents=True, exist_ok=True)

    fake_unsloth = ModuleType("unsloth")

    class FastLanguageModel:
        @staticmethod
        def from_pretrained(**kwargs):
            return _FakeModel(), _FakeTokenizer()

        @staticmethod
        def get_peft_model(model, **kwargs):
            return model

    fake_unsloth.FastLanguageModel = FastLanguageModel
    monkeypatch.setitem(sys.modules, "unsloth", fake_unsloth)

    def _fake_train(self, model, tokenizer, preference_data, config, **kwargs):
        out = Path(kwargs.get("output_dir", output_dir))
        ckpt = out / "dpo_checkpoint"
        ckpt.mkdir(parents=True, exist_ok=True)
        (ckpt / "adapter_config.json").write_text(
            json.dumps({
                "peft_type": "LORA",
                "base_model_name_or_path": config["model_name_or_path"],
            }),
            encoding="utf-8",
        )
        return str(ckpt)

    monkeypatch.setattr(UnslothBackend, "load_model", lambda self, config: (_FakeModel(), _FakeTokenizer()))
    monkeypatch.setattr(UnslothBackend, "train", _fake_train)

    monkeypatch.setattr(
        sys,
        "argv",
        ["train.py", "--config", str(config_path), "--output-dir", str(output_dir)],
    )

    train_script.main()

    ckpt = output_dir / "dpo_checkpoint" / "adapter_config.json"
    assert ckpt.is_file()
    adapter_meta = json.loads(ckpt.read_text(encoding="utf-8"))
    assert adapter_meta["peft_type"] == "LORA"

    peft_module = ModuleType("peft")

    class _PeftModel:
        @staticmethod
        def from_pretrained(base, path):
            return SimpleNamespace(metadata_path=path)

    peft_module.PeftModel = _PeftModel
    transformers_module = ModuleType("transformers")

    class _AutoTokenizer:
        @staticmethod
        def from_pretrained(*args, **kwargs):
            tok = SimpleNamespace()
            tok.pad_token = None
            tok.eos_token = "<eos>"
            return tok

    class _AutoModel:
        @staticmethod
        def from_pretrained(*args, **kwargs):
            return SimpleNamespace()

    transformers_module.AutoTokenizer = _AutoTokenizer
    transformers_module.AutoModelForCausalLM = _AutoModel
    torch_module = ModuleType("torch")
    torch_module.bfloat16 = "bfloat16"

    monkeypatch.setitem(sys.modules, "peft", peft_module)
    monkeypatch.setitem(sys.modules, "transformers", transformers_module)
    monkeypatch.setitem(sys.modules, "torch", torch_module)

    result = UnifiedModelLoader({
        "model_name_or_path": "dummy-local-model",
        "adapter_path": str(output_dir / "dpo_checkpoint"),
    }).load()
    assert result.source == "peft_adapter"
    assert result.metadata["peft_type"] == "LORA"


def test_get_trainer_backend_unsloth():
    from src.training.backends import get_trainer_backend

    backend = get_trainer_backend("unsloth")
    assert backend.name == "unsloth"
