# Scenario Reasoner LM

A training suite for open-source reasoning LLMs that learn to reason through
structured causal and counterfactual scenarios. The repository includes scenario
generation, datasets, metrics, reward composition, DPO training scaffolding,
robustness evaluation, monitoring, and experiment logging.

## Getting Started

```bash
pip install -r requirements.txt
pytest tests/
```

Generate a small causal scenario set:

```bash
python scripts/generate_scenarios.py \
  --chain-lengths 3 \
  --domains physical \
  --difficulties easy \
  --intervention-types direct \
  --n-per-combo 10 \
  --output data/raw/causal/test_scenarios.jsonl
```

Run training and evaluation from the causal RLHF config:

```bash
python scripts/train.py \
  --config experiments/configs/causal_rlhf_config.json \
  --output-dir experiments/results/causal_rlhf_run_01

python scripts/evaluate.py \
  --config experiments/configs/causal_rlhf_config.json \
  --checkpoint experiments/results/causal_rlhf_run_01/dpo_checkpoint \
  --output-dir experiments/results/causal_rlhf_run_01/eval \
  --n-eval 50
```

## Components

### Scenarios
- `src/scenarios/base_scenario.py` - Generic scenario abstraction and instance model.
- `src/scenarios/causal/taxonomy.py` - Causal theta parameter space and grid sampler.
- `src/scenarios/causal/generator.py` - Template-based causal/counterfactual scenario generation.
- `src/scenarios/causal/templates.py` - Domain entities and prompt/reasoning templates.

### Data
- `src/data/base_dataset.py` - Abstract base dataset.
- `src/data/custom_datasets.py` - Custom PyTorch scenario and reasoning-trace datasets.
- `src/data/causal_dataset.py` - Causal reasoning dataset with theta metadata.
- `src/data/hf_datasets.py` - HuggingFace dataset/model wrappers.
- `src/data/evaluators.py` - Dataset quality and coverage utilities.

### Training And Evaluation
- `src/models/model_wrapper.py` - Deferred-import QLoRA/LoRA model loader.
- `src/training/causal_reward.py` - Rule-based causal task reward.
- `src/training/reward_composer.py` - Task, CoT, ToT, and Aha reward composition.
- `src/training/preference_builder.py` - DPO preference-pair construction.
- `src/training/rlhf_trainer.py` - TRL DPO trainer wrapper.
- `src/evaluation/robustness_eval.py` - Per-theta robustness evaluation and JSON reporting.

### Metrics, Losses, Monitoring, Logging
- `src/metrics/base_metrics.py` and `src/metrics/causal_metrics.py` - Metric registry and causal metrics.
- `src/losses/base_losses.py` - Token-level and sequence-level loss bases.
- `src/monitoring/` - CoT, ToT, and Aha pattern monitors.
- `src/logging/local_logger.py` - Structured local log and JSONL metrics writer.
- `src/logging/wandb_logger.py` - Optional Weights & Biases wrapper.

## Portfolio Results

Baseline result rows must identify the model, config, theta grid, evaluation
sample count, seed, checkpoint/source, and report path. Metrics should come from
`robustness_report.json` aggregate values. Use `TBD` for cells that have not
been run or verified yet.

| Experiment | Model | Config | Theta Grid | N Eval | Seed | Causal Chain Accuracy | Counterfactual Validity | Trajectory Consistency | Report |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Baseline causal RLHF | `mistralai/Mistral-7B-Instruct-v0.2` | `experiments/configs/causal_rlhf_config.json` | chain lengths `[3, 5]`; interventions `direct`, `counterfactual`; domains `physical`, `social`; difficulties `easy`, `medium` | `50` | `42` | TBD | TBD | TBD | TBD |

## Project Structure

See [PROJECT_STRUCTURE.md](PROJECT_STRUCTURE.md) for the complete folder layout.

## License

Apache License 2.0 - Copyright 2026 artaasd95
