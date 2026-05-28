# Scenario Reasoner LM

A training suite for open-source reasoning LLMs that learn to reason through
structured causal and counterfactual scenarios. The repository includes scenario
generation, datasets, metrics, reward composition, DPO training scaffolding,
robustness evaluation, monitoring, and experiment logging.

## Strategic Direction

The long-term showcase direction is to package this foundation as an
**Enterprise Risk Scenario Generator**: a system that ingests a 10-K financial
filing, extracts risk evidence, and generates five source-grounded catastrophic
risk scenarios with auditable reasoning traces.

The current codebase provides the causal scenario, reward, monitoring, training,
and robustness-evaluation substrate. The next product layer should add 10-K
ingestion, DSPy reasoning modules and optimizers, Langfuse white-box traces, a
simple Streamlit or Gradio UI, Dockerized environments, and a focused demo path.

See [Enterprise Risk Demo Contract](docs/enterprise-risk-demo.md) for the bundled
10-K showcase scope, non-goals, and decision log.

### Enterprise risk demo (S2)

**Goal:** five catastrophic-but-plausible enterprise risk scenarios from one bundled 10-K.

**Non-goals:** no financial advice, no broad GRC platform, no training-first detour.

```bash
pip install -r requirements.txt -r requirements-enterprise.txt
python scripts/run_enterprise_demo.py --offline
python scripts/export_demo_artifacts.py --output artifacts/enterprise_demo
streamlit run src/ui/streamlit_app.py
```

Bundled sample filing: `data/samples/tenk/acme_corp_10k.txt`  
Tracing: [docs/langfuse-tracing.md](docs/langfuse-tracing.md) · Env template: `.env.example`

### Enterprise eval and optimizers (S4)

Eval rubric, baseline gates, and optimizer comparison scaffolding (no live API runs required in dev):

- [docs/eval-enterprise-risk.md](docs/eval-enterprise-risk.md) — S2 criteria rubric and decision log
- [docs/eval/baseline_scores.json](docs/eval/baseline_scores.json) — checked-in regression floors
- [docs/eval/results/bootstrap_fewshot/](docs/eval/results/bootstrap_fewshot/) — BootstrapFewShot baseline report (JSON + MD)

```bash
# When API budget is reserved (offline default needs no keys):
python scripts/run_enterprise_eval.py --offline
python scripts/compare_enterprise_optimizers.py --dry-run
```

MIPRO: set `ENABLE_MIPRO=1` and `--optimizer MIPRO` on the demo runner; falls back to BootstrapFewShot on failure.  
CI regression workflow template (manual/disabled): `.github/workflows/enterprise_eval_regression.yml`

### Scenario search extensions (S6)

Game-theoretic staged action vectors, algorithm cards (node / action / operator),
search-graph node monitoring, and financial / market-making reasoning θ — without
changing the headline 10-K benchmark.

- [docs/project-track.md](docs/project-track.md) — single-page orientation
- [docs/scenario-search-extensions-contract.md](docs/scenario-search-extensions-contract.md) — S6 contract and phasing
- ADRs: `docs/adr/game-theoretic-action-space.md`, `algorithm-cards-operators.md`, `search-node-monitoring.md`, `financial-risk-and-market-making.md`
- Code: `src/search/`, `src/scenarios/financial/`

```bash
pytest tests/unit/test_search_extensions.py -v
```

### Scenario simulation and measurement (S5)

Simulation (θ → world → trace) vs measurement (scores, θ slices). Two path modes: **wide** (grid / Monte Carlo, many paths) and **bounded** (fixed stages, e.g. five enterprise cards).

- [docs/scenario-simulation-paths.md](docs/scenario-simulation-paths.md) — contract, decision table, resource gates
- [docs/adr/simulation-vs-measurement.md](docs/adr/simulation-vs-measurement.md) — ADR

**Dev/CI default:** mock/smoke only; live runs require `ALLOW_LIVE_PROVIDER=1`. Until the full S5 sprint pipeline, run **unit tests** (not full CLI pipelines in CI):

```bash
pytest tests/unit/test_scenario_simulation_runner.py \
       tests/unit/test_goal_preservation_metrics.py \
       tests/unit/test_scenario_measurement.py \
       tests/integration/test_reasoning_path_audit_smoke.py -v
```

Optional local artifact generation (no network):

```bash
python scripts/run_scenario_simulation.py
python scripts/run_scenario_measurement.py --smoke --output docs/eval/results/scenario_measurement
```

Docker (CPU smoke, offline):

```bash
docker compose up enterprise-demo
```

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

### Enterprise risk demo (10-K)
- `docs/enterprise-risk-demo.md` - Demo contract, non-goals, decision log.
- `src/risk/schema.py` - Scenario card and evidence chunk schema.
- `src/risk/enterprise_theta.py` - Enterprise risk Θ (extends causal θ pattern).
- `src/ingestion/` - 10-K load, SEC section extract, evidence chunking.
- `src/dspy_modules/` - DSPy signatures and pipeline modules (offline stubs supported).
- `src/tracing/` - Langfuse `tenk_demo_run` trace contract.
- `src/ui/streamlit_app.py` - Streamlit demo UI.
- `scripts/run_enterprise_demo.py` - CLI demo runner (`--optimizer BootstrapFewShot|MIPRO`).
- `scripts/run_enterprise_eval.py` - Tiny eval scorer (S2 criteria).
- `scripts/compare_enterprise_optimizers.py` - Optimizer comparison (`--dry-run` in dev).
- `src/dspy_modules/eval_metrics.py` - Eval loader and per-criterion scoring.
- `src/dspy_modules/optimize.py` - BootstrapFewShot / MIPRO feature flag.
- `src/eval/enterprise_eval_schema.py` - Stable eval artifact schema.

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
