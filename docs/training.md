# Training Guide

Scenario Reasoner LM supports DPO/RLHF fine-tuning with two backends:

| Backend | Package | Use case |
|---------|---------|----------|
| `trl` (default) | `transformers` + `peft` + `bitsandbytes` | Baseline QLoRA; works on any CUDA GPU |
| `unsloth` | Optional extra `pip install -e ".[unsloth]"` | Faster LoRA iteration; recommended for Qwen portfolio |

## Environment variables

| Variable | Required | Purpose |
|----------|----------|---------|
| `SCENARIO_MODELS_ROOT` | Yes (for Qwen registry) | Local root for pre-downloaded base models. Layout: `{SCENARIO_MODELS_ROOT}/Qwen/Qwen3-0.6B/` |
| `SCENARIO_TRAINED_MODELS_ROOT` | For artifact preservation | Post-train LoRA adapters copied here as `{model_id}/{run_id}/` |

There are **no repo-relative defaults** for these paths. Set them before training or evaluation.

## GPU requirements

| Tier | Example model | VRAM |
|------|---------------|------|
| Smoke | `qwen3-0.6b`, `qwen2.5-0.5b` | ≥8 GB |
| Mid | `qwen2.5-3b`, `qwen2.5-coder-3b` | ≥16 GB |
| Large (gated) | `qwen2.5-7b`, `qwen3-4b` | ≥24 GB with 4-bit QLoRA |

Unsloth requires CUDA 11.8+ or 12.x. Install the matching PyTorch build before the optional extra.

## Quick start

```bash
export SCENARIO_MODELS_ROOT=/path/to/models
export SCENARIO_TRAINED_MODELS_ROOT=/path/to/trained

# Cache models if missing locally
python scripts/download_models.py --model-id qwen3-0.6b

# Unsloth DPO (smoke config)
pip install -e ".[unsloth]"
python scripts/train.py --config configs/training/unsloth_dpo_example.yaml

# Vanilla TRL backend (legacy JSON config)
python scripts/train.py --config experiments/configs/causal_rlhf_config.json
```

## Config keys

```yaml
model_id: qwen3-0.6b          # resolves via configs/models/qwen_portfolio.yaml
# OR legacy:
# model_name_or_path: mistralai/Mistral-7B-Instruct-v0.2

training:
  backend: unsloth             # trl | unsloth
  policy: causal_default       # PolicyRegistry reward weights + θ mix
  data_source: inline          # inline | distilled
  distilled_manifest: null     # path when data_source=distilled
  monitor_gates: false
```

Policies (`causal_default`, `mixed_scenario`) are defined in `src/training/policies/registry.py`.

## Throughput / memory notes

Placeholder — fill after first GPU run on one Qwen base:

| Backend | Model | Tokens/s (train) | Peak VRAM | Notes |
|---------|-------|------------------|-----------|-------|
| `trl` | Qwen3-0.6B | TBD | TBD | Vanilla PEFT + bitsandbytes |
| `unsloth` | Qwen3-0.6B | TBD | TBD | FastLanguageModel + Unsloth kernels |

## Distillation

**Decision: yes — required for production-quality DPO beyond smoke.**

| Path | Use |
|------|-----|
| Inline `CausalScenarioGenerator` + `PreferenceBuilder` | Smoke / CI only — limited diversity |
| Distilled JSONL from stronger teacher (Colab) | Primary training path for scenario DPO |

Workflow:

1. Build θ-conditioned traces in `notebooks/distill_teacher_colab.ipynb` (Colab).
2. Version output under `data/distilled/<name>/` with `manifest.json`.
3. Train with `training.data_source: distilled` and `training.distilled_manifest`.

See `data/distilled/README.md` for layout and manifest schema.

## Post-train artifacts

After DPO, when `SCENARIO_TRAINED_MODELS_ROOT` is set, `train.py` promotes the checkpoint to:

```
{SCENARIO_TRAINED_MODELS_ROOT}/{model_id}/{run_id}/
  adapter_config.json
  adapter_model.safetensors (or bin shards)
  tokenizer_config.json
  train_config.json
```

Use `scripts/compare_pre_post_train.py` to measure base vs post-train quality deltas.
