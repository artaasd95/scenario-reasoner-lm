# Data Platform (S9)

**Contract:** DP-2026-02 — θ dataclass normalization at the pipeline normalize stage.

## Overview

The data platform turns scenario/search paths into training and eval artifacts via YAML-configured pipelines:

`normalize → measure → label → split → filter`

| Artifact | Path |
| --- | --- |
| Train config | [configs/data/train.yaml](../configs/data/train.yaml) |
| Eval config | [configs/data/eval.yaml](../configs/data/eval.yaml) |
| Feedback config | [configs/data/feedback.yaml](../configs/data/feedback.yaml) |
| ScenarioPathRow | [src/data/cards/scenario_path_row.py](../src/data/cards/scenario_path_row.py) |
| θ normalization | [src/data/theta_normalization.py](../src/data/theta_normalization.py) |
| Pipeline runner | [src/data/pipeline/runner.py](../src/data/pipeline/runner.py) |
| Sources | [src/data/sources/](../src/data/sources/) |

## θ normalization (DP-2026-02)

Dict payloads from JSONL/fixtures convert to typed θ at **normalize**:

- `CausalTheta`, `GameTheoreticTheta`, `FinancialRiskTheta`, `MarketMakingReasoningTheta`
- Invalid θ is dropped with a logged reason (see `tests/unit/test_theta_normalization.py`)

## Quickstart

```bash
# Build training artifact from bundled generator fixtures
python scripts/build_dataset.py --config configs/data/train.yaml

# Eval mode writes scenario_measurement.json stub
python scripts/build_dataset.py --config configs/data/eval.yaml

# Training with data config (smoke: pipeline only, no GPU)
python scripts/train.py --config experiments/configs/causal_rlhf_config.json \
  --data-config configs/data/train.yaml --output-dir experiments/results/smoke

# Feedback stub
python scripts/apply_feedback.py --config configs/data/feedback.yaml
```

Stage drop/enrich counts are logged in the pipeline `stats` object (`loaded`, `normalized`, `measured`, `dropped`, etc.).

## Measurability

Eval/train measurement keys align with [scenario-measurability-contract.md](scenario-measurability-contract.md).

## Related

- [scenario-simulation-paths.md](scenario-simulation-paths.md)
- [project-track.md](project-track.md)
