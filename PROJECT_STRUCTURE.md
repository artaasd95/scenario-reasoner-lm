# Scenario Reasoner LM - Project Structure

Updated for S10 milestone: serving API, verification, expert feedback, data pipeline wiring.

## New packages (S10)

```text
src/serving/           # FastAPI schemas, providers, scenario_service
src/verification/      # numerical_verifier, local_engine, report
src/feedback/          # expert_schema, JSONL store
apps/review_ui.py      # Streamlit expert review UI
```

## Key scripts

| Script | Purpose |
|--------|---------|
| `scripts/train.py` | DPO training; `--data-config` loads pipeline JSONL |
| `scripts/evaluate.py` | Robustness eval + `scenario_measurement.json` |
| `scripts/serve.py` | FastAPI server CLI |
| `scripts/generate.py` | Offline ScenarioArtifact JSON |
| `scripts/apply_feedback.py` | Feedback JSONL → preference pairs / policy weights |
| `scripts/build_dataset.py` | Data-platform pipeline runner |

## Data pipeline (S9/S10)

```text
src/data/pipeline/
  config.py            # YAML pipeline config loader
  runner.py            # normalize → measure → label → split → filter
  source_factory.py    # generator, fixture sources
configs/data/
  train.yaml           # → data/processed/train/pipeline_train.jsonl
  eval.yaml
  feedback.yaml
```

## Training (S10 wiring)

```text
src/training/policies/registry.py     # PolicyRegistry + YAML policies
src/training/callbacks/               # RewardDecomposition, ThetaStratifiedEarlyStop
src/training/train_helpers.py         # resolve_training_config, pipeline JSONL loader
src/models/loaders/unified.py         # UnifiedModelLoader (hub, checkpoint, PEFT)
experiments/configs/policies/         # default_causal.yaml, mixed_scenario.yaml
```

## Serving & deploy

```text
configs/serve.yaml
Dockerfile.serve                      # Multi-stage API image
docker-compose.yml                    # scenario-api (default) + enterprise-demo (profile demo)
docker-compose.override.yml           # Local dev mounts + review-ui profile
docs/serving.md
docs/deployment.md
.github/workflows/ci.yml              # pytest -k smoke, ruff, mypy
```

## Tests (smoke)

```text
tests/integration/test_smoke_train_eval.py
tests/integration/test_pipeline_train_wire.py
tests/integration/test_service_smoke.py
tests/unit/test_serving_schemas.py
tests/unit/test_numerical_verifier.py
tests/unit/test_apply_feedback.py
```

See [README.md](README.md) for quickstart commands.
