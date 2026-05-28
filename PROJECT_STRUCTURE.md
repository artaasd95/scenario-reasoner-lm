# Scenario Reasoner LM - Project Structure

This document describes the current repository layout for Scenario Reasoner LM.
The project has moved beyond the initial scaffold: it now includes causal
scenario generation, reward composition, DPO training wiring, robustness
evaluation, logging, and focused smoke tests.

## Complete Structure

```text
scenario-reasoner-lm/
|-- README.md
|-- PROJECT_STRUCTURE.md
|-- requirements.txt
|-- LICENSE
|
|-- src/
|   |-- __init__.py
|   |
|   |-- data/
|   |   |-- __init__.py
|   |   |-- base_dataset.py          # Abstract dataset base class
|   |   |-- custom_datasets.py       # Scenario and reasoning-trace datasets
|   |   |-- causal_dataset.py        # Causal dataset with theta metadata
|   |   |-- evaluators.py            # Dataset quality utilities
|   |   `-- hf_datasets.py           # HuggingFace dataset wrappers
|   |
|   |-- evaluation/
|   |   |-- __init__.py
|   |   `-- robustness_eval.py       # Per-theta robustness report generation
|   |
|   |-- logging/
|   |   |-- __init__.py
|   |   |-- local_logger.py          # Local log and JSONL metrics writer
|   |   `-- wandb_logger.py          # Optional Weights & Biases wrapper
|   |
|   |-- losses/
|   |   |-- __init__.py
|   |   `-- base_losses.py           # Token and sequence loss bases
|   |
|   |-- metrics/
|   |   |-- __init__.py
|   |   |-- base_metrics.py          # Metric base class and registry
|   |   `-- causal_metrics.py        # Causal reasoning metrics
|   |
|   |-- models/
|   |   |-- __init__.py
|   |   `-- model_wrapper.py         # Deferred-import QLoRA/LoRA loader
|   |
|   |-- monitoring/
|   |   |-- __init__.py
|   |   |-- aha_monitor.py
|   |   |-- cot_monitor.py
|   |   `-- tot_monitor.py
|   |
|   |-- search/                      # S6: cards, game θ, graph monitor, manifold
|   |   |-- __init__.py
|   |   |-- cards.py                 # NodeCard, ActionCard, AlgorithmCard
|   |   |-- game_theta.py            # GameTheoreticTheta (staged action vectors)
|   |   |-- graph_monitor.py         # SearchGraphMonitor
|   |   `-- manifold.py              # ActionManifold (box / simplex)
|   |
|   |-- scenarios/
|   |   |-- __init__.py
|   |   |-- base_scenario.py         # Generic scenario abstraction
|   |   |-- theta_mapping.py         # Enterprise / causal / S6 θ maps
|   |   |-- simulation_runner.py     # S5 simulation fixtures
|   |   |-- financial/               # S6 financial + market-making θ
|   |   |   |-- financial_risk_theta.py
|   |   |   `-- market_making_theta.py
|   |   `-- causal/
|   |       |-- __init__.py
|   |       |-- generator.py          # Causal/counterfactual generator
|   |       |-- taxonomy.py           # Causal theta dataclass and sampler
|   |       `-- templates.py          # Domain and prompt templates
|   |
|   `-- training/
|       |-- __init__.py
|       |-- causal_reward.py         # Rule-based causal reward
|       |-- preference_builder.py    # DPO pair construction
|       |-- reward_composer.py       # Task + monitor reward composition
|       `-- rlhf_trainer.py          # TRL DPO trainer wrapper
|
|-- scripts/
|   |-- evaluate.py                  # Robustness evaluation CLI
|   |-- generate_scenarios.py        # Scenario JSONL generation CLI
|   `-- train.py                     # Causal RLHF training CLI
|
|-- experiments/
|   |-- configs/
|   |   |-- causal_rlhf_config.json
|   |   `-- example_config.json
|   `-- results/
|
|-- data/
|   |-- raw/
|   `-- processed/
|
|-- tests/
|   |-- __init__.py
|   |-- unit/
|   |   |-- __init__.py
|   |   |-- test_datasets.py
|   |   |-- test_logging.py
|   |   |-- test_losses.py
|   |   |-- test_metrics.py
|   |   |-- test_monitoring.py
|   |   `-- test_scenarios.py
|   `-- integration/
|       |-- __init__.py
|       |-- test_pipeline.py
|       `-- test_train_eval_smoke.py
|
`-- docs/
    |-- project-track.md
    |-- scenario-search-extensions-contract.md
    |-- detailed-improvement-roadmap.md
    |-- ideas-plan.md
    |-- scenario-search-formulation.md
    |-- scenario-simulation-paths.md
    |-- enterprise-risk-demo.md
    `-- adr/
        |-- simulation-vs-measurement.md
        |-- game-theoretic-action-space.md
        |-- algorithm-cards-operators.md
        |-- search-node-monitoring.md
        `-- financial-risk-and-market-making.md
```

## Key Components

### Causal Scenario Stack
- `src/scenarios/causal/taxonomy.py` defines `CausalTheta` and grid sampling.
- `src/scenarios/causal/generator.py` turns theta values into prompts, reasoning traces, answers, and metadata.
- `src/data/causal_dataset.py` carries generated scenario data into training and exposes theta metadata as plain dictionaries.

### Training Stack
- `src/training/causal_reward.py` scores causal reasoning traces without a learned reward model.
- `src/training/reward_composer.py` combines task reward with CoT, ToT, and Aha monitoring signals.
- `src/training/preference_builder.py` builds DPO chosen/rejected pairs from generated candidates.
- `src/training/rlhf_trainer.py` wraps TRL `DPOTrainer` for checkpointing and reporting.
- `scripts/train.py` wires generation, model loading, preference construction, DPO training, and local/W&B logging.

### Evaluation Stack
- `src/metrics/causal_metrics.py` provides causal chain accuracy, counterfactual validity, and trajectory consistency metrics.
- `src/evaluation/robustness_eval.py` evaluates across a theta grid and saves `robustness_report.json`.
- `scripts/evaluate.py` loads a checkpoint, runs robustness evaluation, and logs aggregate metrics.

### Logging And Tests
- `src/logging/local_logger.py` writes human-readable logs plus JSONL config/step/epoch/monitoring records.
- `src/logging/wandb_logger.py` is a safe optional W&B wrapper.
- `tests/unit/test_logging.py` covers the local logger contract and lifecycle.
- `tests/integration/test_train_eval_smoke.py` exercises train/eval CLI wiring with heavyweight model calls patched out.
