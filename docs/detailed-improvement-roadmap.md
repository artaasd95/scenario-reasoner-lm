# Scenario Reasoner LM - Detailed Improvement Roadmap

## Purpose and Scope

This document replaces the prior high-level suggestion map with a verified, implementation-oriented roadmap aligned to the current repository state.

Goals:
- Verify what is already implemented vs. what remains aspirational.
- Define a phased, execution-ready improvement map.
- Specify actions, file-level impact, tests, acceptance criteria, and sequencing.
- Provide project-management visibility for delivery planning and risk control.

---

## Verified Current State (Codebase Audit)

### What is implemented and usable

- Scenario representation and generation:
  - `src/scenarios/base_scenario.py`
  - `src/scenarios/causal/taxonomy.py`
  - `src/scenarios/causal/generator.py`
  - `src/scenarios/causal/templates.py`
- Causal dataset support:
  - `src/data/causal_dataset.py`
- Monitoring for reasoning signals:
  - `src/monitoring/cot_monitor.py`
  - `src/monitoring/tot_monitor.py`
  - `src/monitoring/aha_monitor.py`
- Reward and RLHF building blocks:
  - `src/training/causal_reward.py`
  - `src/training/reward_composer.py`
  - `src/training/preference_builder.py`
  - `src/training/rlhf_trainer.py`
- Metrics and robustness evaluation:
  - `src/metrics/base_metrics.py`
  - `src/metrics/causal_metrics.py`
  - `src/evaluation/robustness_eval.py`
- Logging infrastructure:
  - `src/logging/local_logger.py`
  - `src/logging/wandb_logger.py`
- Unit and light integration tests:
  - `tests/unit/*`
  - `tests/integration/test_pipeline.py`

### Verified implementation gaps and risks

- Training/evaluation scripts call a non-existent logger API (`log_metrics`), while `LocalLogger` exposes `log_step`, `log_epoch`, and `log_monitoring`:
  - `scripts/train.py`
  - `scripts/evaluate.py`
  - `src/logging/local_logger.py`
- Script stability issues in `scripts/train.py`:
  - Duplicate `if __name__ == "__main__":` blocks.
  - `local_logger` and `wandb_logger` are referenced outside `main()`.
- Theta typing inconsistency risk:
  - Dataset emits `theta` as dict via `CausalReasoningDataset.__getitem__`.
  - Reward function expects attribute access on `theta` in `src/training/causal_reward.py`.
- No benchmark adapters for external reasoning datasets.
- No formal verification engine (only heuristic/regex checks).
- No CI pipeline for regression protection.
- Documentation (`PROJECT_STRUCTURE.md`) is stale relative to current modules.

---

## Alignment Against Previous Suggestion Themes

### Fully or partially aligned now

- Formal scenario representation: partially aligned (causal theta and scenario interfaces exist).
- Evaluation framework: partially aligned (causal metrics + robustness evaluator exist).
- Structured monitoring/observability: partially aligned (monitor classes + local/W&B logging exist).
- RLHF-style pipeline: partially aligned (preference builder and trainer exist).

### Not yet aligned

- External benchmark suite (bAbI, CLUTRR, ProofWriter, LogicGrid adapters).
- Formal consistency verification and contradiction proving.
- Multiple active reasoning strategies (ToT search, GoT, RAP, self-consistency decoding).
- Hallucination and calibration metrics.
- Production engineering baseline (CI, runbooks, reproducibility constraints, release checks).

### Better approach than the original suggestions

- Prioritize hardening the existing causal stack before adding many new research tracks.
- Treat monitors as first-class reward/evaluation signals only after measurement reliability is validated.
- Add formal verification incrementally with graph constraints before model checking tools.
- Defer speculative domains (legal/scientific/multi-agent) until core benchmark and reliability gates are passing.

---

## Delivery Model (Project Management View)

### Workstreams

- WS1: Pipeline Reliability and Runtime Correctness
- WS2: Evaluation and Benchmark Expansion
- WS3: Verification and Reasoning Quality Controls
- WS4: Operations, Reproducibility, and CI
- WS5: Advanced Research Differentiation

### Cadence and governance

- Sprint length: 2 weeks
- Release checkpoints:
  - R1: Foundation stable (end of Sprint 1)
  - R2: Evaluation/verification baseline (end of Sprint 3)
  - R3: Advanced reasoning experiments (end of Sprint 5+)
- Mandatory Definition of Done for any feature:
  - Unit tests and integration coverage updated.
  - Documentation updated (`README.md` and relevant docs).
  - Passes CI test workflow.
  - Acceptance criteria in this roadmap satisfied.

### Suggested ownership

- Tech Lead: architecture decisions, milestone sign-off.
- ML Engineer: training pipeline, reward, reasoning strategies.
- Evaluation Engineer: metrics, benchmark adapters, reporting.
- MLOps Engineer: CI, reproducibility, artifacts, runbooks.
- QA/Reviewer: test plan quality and release gate checks.

---

## Phase Plan and Detailed Action Map

## Phase 0 - Stabilize Runtime Foundations (Weeks 1-2)

### Objective

Make train/evaluate workflows executable and test-protected.

### Actions

1) Fix logger contract mismatch
- Files:
  - `scripts/train.py`
  - `scripts/evaluate.py`
  - `src/logging/local_logger.py` (optional compatibility helper)
- Action:
  - Replace `log_metrics(...)` calls with `log_step(...)`/`log_epoch(...)`, or implement a compatibility `log_metrics` wrapper.

2) Fix script entrypoint and lifecycle logic
- File:
  - `scripts/train.py`
- Action:
  - Keep a single `if __name__ == "__main__"` block.
  - Ensure logger closing is inside `main()` and guarded in `try/finally`.

3) Resolve theta type compatibility
- Files:
  - `src/training/causal_reward.py`
  - `src/data/causal_dataset.py` (if needed)
- Action:
  - Accept both dict and dataclass-like theta in reward scoring paths.

4) Add smoke integration test for train/eval path
- Files:
  - `tests/integration/test_train_eval_smoke.py` (new)
  - Optional fixtures under `tests/fixtures/`
- Action:
  - Mock lightweight model/tokenizer behavior and verify end-to-end control flow.

5) Update docs to match actual state
- Files:
  - `README.md`
  - `PROJECT_STRUCTURE.md`
- Action:
  - Remove stale "scaffold only" wording where no longer true.
  - Add known limitations and minimal run commands.

### Tests

- Unit:
  - `tests/unit/test_logging.py` (new: logger method compatibility and output records)
  - `tests/unit/test_causal_reward.py` update/add cases for dict theta and object theta.
- Integration:
  - `tests/integration/test_train_eval_smoke.py` (new)

### Acceptance criteria

- `scripts/train.py` runs through scenario generation, preference creation, and trainer invocation without method errors.
- `scripts/evaluate.py` produces `robustness_report.json` with aggregate metrics.
- No duplicate entrypoint blocks in train script.
- New tests pass locally: `pytest tests/unit tests/integration`.

---

## Phase 1 - Evaluation and Benchmark Baseline (Weeks 3-6)

### Objective

Create reliable performance baselines across internal and external reasoning datasets.

### Actions

1) Add benchmark adapters with unified schema
- New module:
  - `src/data/benchmarks/`
    - `babi.py`
    - `clutrr.py`
    - `proofwriter.py`
    - `logicgrid.py`
    - `registry.py`
- Action:
  - Normalize to a common sample shape:
    - `input`
    - `reasoning_trace` (optional)
    - `output`
    - `metadata` (dataset name, split, difficulty)

2) Extend evaluate script for benchmark mode
- File:
  - `scripts/evaluate.py`
- Action:
  - Add CLI flags for benchmark dataset selection and report slicing by dataset and difficulty.

3) Introduce reporting utilities
- New module:
  - `src/evaluation/reporting.py`
- Action:
  - Generate JSON summaries with:
    - overall metrics
    - per-dataset
    - per-theta
    - failure buckets

### Tests

- Unit:
  - `tests/unit/test_benchmark_adapters.py` (new)
  - `tests/unit/test_reporting.py` (new)
- Integration:
  - `tests/integration/test_benchmark_eval_smoke.py` (new)

### Acceptance criteria

- Each benchmark adapter loads a fixture sample and maps to normalized schema.
- Evaluate script can run with at least one external benchmark.
- Report includes aggregate and stratified sections.

---

## Phase 2 - Verification and Quality Controls (Weeks 7-10)

### Objective

Raise reasoning reliability with explicit consistency checks and quality metrics.

### Actions

1) Add graph-based consistency verifier
- New module:
  - `src/verification/consistency.py`
- Action:
  - Build causal graph from trace/metadata and validate:
    - cycle detection
    - contradiction constraints
    - intervention coherence

2) Add calibration and hallucination metrics
- New modules:
  - `src/metrics/calibration_metrics.py`
  - `src/metrics/hallucination_metrics.py`
- Action:
  - Add confidence proxy and unsupported-claim scoring heuristics.

3) Integrate verifier outputs into reward composition (optional gate)
- Files:
  - `src/training/reward_composer.py`
  - `src/training/causal_reward.py`
- Action:
  - Add weighted verifier penalty/bonus and keep weight configurable.

### Tests

- Unit:
  - `tests/unit/test_verification_consistency.py` (new)
  - `tests/unit/test_calibration_metrics.py` (new)
  - `tests/unit/test_hallucination_metrics.py` (new)
- Integration:
  - `tests/integration/test_reward_verifier_integration.py` (new)

### Acceptance criteria

- Verifier flags at least one synthetic inconsistent trace and passes a valid trace.
- New metrics are available through metric registry.
- Reward composer can include verifier signal without breaking training flow.

---

## Phase 3 - Operations and Reproducibility (Weeks 11-12)

### Objective

Establish merge-safe and experiment-reproducible engineering practices.

### Actions

1) Add CI workflow
- New file:
  - `.github/workflows/test.yml`
- Action:
  - Run lint (if configured), unit tests, integration smoke tests.

2) Add runbook and experiment manifest
- New docs:
  - `docs/runbook.md`
  - `docs/experiment-governance.md`
- Action:
  - Define artifact naming, seed control, checkpoint retention, and rollback procedure.

3) Add config validation utility
- New script:
  - `scripts/validate_config.py`
- Action:
  - Validate required keys and types before train/eval runs.

### Tests

- Unit:
  - `tests/unit/test_config_validation.py` (new)
- CI checks:
  - Validate workflow executes and gates merge.

### Acceptance criteria

- Pull requests require CI pass before merge.
- Re-running same config/seed yields reproducible report envelope (within tolerance).
- Runbook can be followed by a new contributor end-to-end.

---

## Phase 4 - Advanced Reasoning Strategy Expansion (Weeks 13+)

### Objective

Move from monitoring-only strategy detection to active multi-strategy reasoning.

### Actions

1) Implement strategy modules
- New module:
  - `src/reasoning/`
    - `cot_decoder.py`
    - `tot_search.py`
    - `self_consistency.py`
    - `verifier_guided.py`

2) Add strategy selection in train/eval interfaces
- Files:
  - `scripts/train.py`
  - `scripts/evaluate.py`
  - config files under `experiments/configs/`

3) Add strategy ablation harness
- New module:
  - `src/evaluation/ablation.py`

### Tests

- Unit:
  - `tests/unit/test_reasoning_strategies.py` (new)
- Integration:
  - `tests/integration/test_strategy_ablation_smoke.py` (new)

### Acceptance criteria

- At least two active reasoning strategies are executable from config.
- Ablation report compares strategy-level outcomes on same data slice.
- Strategy behavior is measurable via existing monitors and metrics.

---

## Prioritized Backlog (Execution Order)

1. P0: Fix `train.py` runtime correctness and logger API usage.
2. P0: Fix `evaluate.py` logger API usage and reporting stability.
3. P0: Resolve theta typing compatibility in reward path.
4. P0: Add train/eval smoke integration tests.
5. P1: Refresh docs (`README.md`, `PROJECT_STRUCTURE.md`) to current architecture.
6. P1: Implement benchmark adapters and benchmark evaluation mode.
7. P1: Add reporting module and richer stratified outputs.
8. P1: Implement verification module and wire into evaluation.
9. P2: Add hallucination/calibration metrics.
10. P2: Add CI workflow and runbook/repro governance.
11. P3: Implement active multi-strategy reasoning modules and ablation.
12. P3: Explore formal-method bridges and domain expansions.

---

## Risk Register and Mitigations

- Risk: Script-level breakage blocks iteration.
  - Mitigation: Phase 0 fixes + smoke tests as merge gate.
- Risk: Reward shaping drifts from true reasoning quality.
  - Mitigation: isolate verifier/calibration metrics and compare against held-out benchmark behavior.
- Risk: Benchmark adapters produce inconsistent schemas.
  - Mitigation: strict adapter contract tests and registry checks.
- Risk: Fast feature expansion causes architecture drift.
  - Mitigation: enforce DoD and architecture review at each phase release checkpoint.

---

## KPIs by Phase

- Phase 0:
  - Train/eval script success rate: 100 percent on smoke path.
  - Runtime method errors: 0.
- Phase 1:
  - External benchmark coverage: at least 2 adapters enabled.
  - Per-dataset reporting completeness: 100 percent.
- Phase 2:
  - Consistency violation detection precision/recall on synthetic fixtures: defined and tracked.
  - Calibration/hallucination metrics present in reports.
- Phase 3:
  - CI pass rate and mean time to detect regressions tracked.
  - Reproducibility checks green for fixed seeds.
- Phase 4:
  - Strategy ablation report generated per release candidate.
  - Improvement over baseline metrics demonstrated for at least one strategy.

---

## Definition of Ready for Any New Feature

- Problem statement and expected value are documented.
- File-level design impact is identified.
- Test plan includes unit and at least one integration path.
- Acceptance criteria are explicit and measurable.
- Rollback path is defined for pipeline-impacting changes.

---

## Suggested Immediate Next Actions (This Week)

1) Implement and merge Phase 0 fixes.
2) Add smoke tests and make them pass in CI (or local gate if CI not yet added).
3) Refresh docs to remove stale scaffold statements.
4) Open backlog issues for Phase 1 benchmark and reporting modules with this roadmap as reference.

