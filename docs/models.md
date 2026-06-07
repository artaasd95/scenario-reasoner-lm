# Qwen Portfolio Model Registry

Canonical base models for scenario eval and DPO training. Defined in
[`configs/models/qwen_portfolio.yaml`](../configs/models/qwen_portfolio.yaml).

## Environment

```bash
export SCENARIO_MODELS_ROOT=/path/to/your/models
export SCENARIO_TRAINED_MODELS_ROOT=/path/to/trained/adapters
```

`SCENARIO_MODELS_ROOT` is **required** when using `model_id` in training or eval configs.
There is no repo-relative default.

Expected layout (HuggingFace hub id as subdirectory path):

```
$SCENARIO_MODELS_ROOT/
  Qwen/
    Qwen3-0.6B/
    Qwen3-4B/
    Qwen3-4B-Instruct-2507/
  Qwen2.5-0.5B/
  Qwen2.5-3B/
  Qwen2.5-7B/
  Qwen2.5-Coder-3B/
```

## Canonical models

| model_id | Hub id | Tier |
|----------|--------|------|
| `qwen3-4b-instruct-2507` | Qwen/Qwen3-4B-Instruct-2507 | large |
| `qwen3-4b` | Qwen/Qwen3-4B | large |
| `qwen3-0.6b` | Qwen/Qwen3-0.6B | smoke |
| `qwen2.5-coder-3b` | Qwen/Qwen2.5-Coder-3B | mid |
| `qwen2.5-0.5b` | Qwen/Qwen2.5-0.5B | smoke |
| `qwen2.5-3b` | Qwen/Qwen2.5-3B | mid |
| `qwen2.5-7b` | Qwen/Qwen2.5-7B | large |

## Tiers

| Tier | Models | Use |
|------|--------|-----|
| **smoke** | `qwen3-0.6b`, `qwen2.5-0.5b` | CI, fast iteration, Unsloth smoke |
| **mid** | `qwen2.5-coder-3b`, `qwen2.5-3b` | Benchmark runs |
| **large_gated** | `qwen3-4b`, `qwen3-4b-instruct-2507`, `qwen2.5-7b` | Full comparison; require explicit `--allow-large` in download script or config flag |

## Download / verify

```bash
export SCENARIO_MODELS_ROOT=/path/to/models

python scripts/download_models.py --model-id qwen3-0.6b
python scripts/download_models.py --tier smoke
python scripts/download_models.py --all
```

Skips download when the local directory already exists.

## Usage in configs

```yaml
model_id: qwen3-0.6b
```

Resolved at runtime to `{SCENARIO_MODELS_ROOT}/Qwen/Qwen3-0.6B` via
[`src/models/model_registry.py`](../src/models/model_registry.py).

Legacy `model_name_or_path` (e.g. Mistral) still works without the registry.
