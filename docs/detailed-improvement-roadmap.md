# Scenario Reasoner LM - Detailed Improvement Roadmap

## Purpose and Scope

This document replaces the prior high-level suggestion map with a verified, implementation-oriented roadmap aligned to the current repository state.

Goals:
- Verify what is already implemented vs. what remains aspirational.
- Define a phased, execution-ready improvement map.
- Specify actions, file-level impact, tests, acceptance criteria, and sequencing.
- Provide project-management visibility for delivery planning and risk control.
- **Portfolio lens:** separate a credible public repo from a **hire-me** signal—reproducible benchmark numbers, a defensible “why this wins,” and minimal scope creep.

---

## Portfolio Strategy: “Good Portfolio” vs “Hire Me Immediately”

Use this split to decide **what to build next**. The default should be **fewer surface features** and **more defensible measurements**. Extra modules do not beat a single sharp result with a clear story.

### Good portfolio (credible, professional)

**What a reviewer sees:** clean structure, tests, README that runs, honest limitations.

| Dimension | Target |
|-----------|--------|
| Repo hygiene | README with install + one command that produces an artifact (log, JSON report, or notebook output). |
| Honesty | Explicit scope: causal scenario training/eval, not “general AGI reasoning.” |
| Evidence | At least **one** fixed evaluation protocol (same prompts, same decoding, same seed) with **reported numbers** (even if only on internal/synthetic scenarios first). |
| Reproducibility | Pinned dependencies or lockfile note; config JSON checked in; “how to rerun” in docs. |
| Story | One paragraph: **problem → method → what you measured → outcome.** |

**Acceptance (portfolio bar):** A stranger can clone, run tests, run eval (or smoke eval), and read one table of metrics without trusting your prose alone.

### Hire me immediately (differentiated, interview-ready)

**What a strong hiring manager sees:** **numbers on named benchmarks**, comparison to **stated baselines**, and **why your approach should win** under those constraints—not a longer feature list.

| Dimension | Target |
|-----------|--------|
| Benchmark claim | **1–2 public benchmarks maximum** for the primary claim (e.g. one causal/logical suite you can defend end-to-end). Depth beats breadth. |
| SOTA framing | **Match or beat** a clearly defined baseline under a **declared budget**: same or comparable base model class, training data scale, inference budget (tokens/decoding), and eval script. |
| Reproducibility contract | Frozen `experiments/configs/*`, commit hash, hardware note, seeds, exact eval command, and **artifact** (JSON report + optional W&B run id). |
| “Why us” | A short, falsifiable thesis (see template below): what inductive bias or training signal you add, and **why** it should improve **that** metric on **that** benchmark. |
| Ablations | **2–3** minimal ablations (e.g. reward off, monitors off, DPO off) on the **same** eval—table in README or `docs/results/`. |
| Optional arXiv / tech report | Use when the story needs **citations, related work, and exact protocol** for skeptical readers—not as a substitute for runnable code. |

**Acceptance (hire-me bar):** README opens with a **results table** (metric, baseline, yours, Δ, setup). A reviewer can rerun eval and get numbers within a documented tolerance. You can explain **why** you win or **why** you tied without hand-waving.

### Anti-goals (portfolio mode)

- Adding “Phase 4” breadth (many strategies, many domains) **before** a single benchmark row is reproducible and explained.
- Claiming SOTA without naming **dataset, split, metric, model, and decoding**.
- Treating more files as progress; **more graphs in one PDF** beats more Python modules.

---

## Evidence Stack: What You Ship (Not What You Code)

Prioritize deliverables that fit on a resume line and in an interview.

| Artifact | Purpose | Hire-me acceptance |
|----------|---------|-------------------|
| **Results table** in README | First screen impression | ≥2 rows: baseline + yours; same metric definitions |
| **Frozen eval config + one command** | Trust | Documented command reproduces table within stated variance |
| **Robustness / θ breakdown** (existing evaluator) | “I measure more than accuracy” | One figure or JSON slice: performance vs. scenario parameters |
| **Ablation mini-table** | Proves the method, not the repo size | Same benchmark; only one knob changes per row |
| **Short technical note** (`docs/technical-note.md`) or **arXiv** | Defensible “why” + related work | Problem, method, experiments, limitations—no marketing |

**Rule:** If it does not improve a **number** in the table or the **clarity** of “why,” defer it.

---

## Benchmark and SOTA Contract (Defensible Comparisons)

### Picking the claim

1. **Primary benchmark (1):** The one your README headline uses (e.g. one external causal/logical dataset fully wired, or a published-style subset with clear license).
2. **Secondary (optional):** Only if it strengthens the same story (e.g. robustness slice), not a second unrelated brag.

### Baselines (required for “hire me”)

For each baseline row, document **all** of:

- Base model id and revision
- Fine-tuning recipe (SFT / DPO / none) and data scale
- Decoding: temperature, max tokens, stop sequences
- Metric definition (exact match, F1, custom causal metric—spell it out)

**Better than vague SOTA:** “We compare to **[named method]** on **[named split]** using **[metric]** because **[reason]**.” Tying or beating a **reproduced** baseline you own is stronger than an unverifiable leaderboard screenshot.

### The “why we win” template (README + interviews)

Fill this in once; reuse everywhere:

1. **Structural assumption:** What structure do you inject (e.g. θ-conditioned scenarios, rule-based reward shaping, monitor-augmented preferences)?
2. **Mechanism:** How that changes the training signal or the optimization landscape.
3. **Prediction:** What should improve first (e.g. counterfactual validity under intervention type X)?
4. **Result:** Point to the row in the table and the ablation that isolates the mechanism.

If (3) and (4) do not align, fix the claim—not the adjective.

---

## Optional: arXiv / Technical Report Path

Use when you need a **single citable PDF** for recruiters, collaborators, or follow-up interviews—not as extra feature work.

**Minimum viable report (4–6 pages):**

1. **Abstract:** One sentence result + one sentence method.
2. **Setup:** Benchmark, splits, metrics, baselines, compute.
3. **Method:** Scenario parameterization (θ), reward/DPO/monitoring—only what you actually run.
4. **Results:** Main table + robustness + ablations.
5. **Limitations:** Where the method fails; no SOTA without caveats.
6. **Reproducibility:** Exact command, config path, commit hash.

**Hire-me check:** Someone can read only the abstract and the main table and still understand the contribution.

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
- **Portfolio:** Prefer **one benchmark headline + ablations** over parallel feature tracks; ship the **evidence stack** (table, config, rerun command) before expanding code surface.

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
- **Portfolio Definition of Done (hire-me track):** Any change that touches training or eval also updates (or explicitly defers) the **README results table**, **baseline row**, and **repro command**—not only internal code.

### Suggested ownership

- Tech Lead: architecture decisions, milestone sign-off.
- ML Engineer: training pipeline, reward, reasoning strategies.
- Evaluation Engineer: metrics, benchmark adapters, reporting.
- MLOps Engineer: CI, reproducibility, artifacts, runbooks.
- QA/Reviewer: test plan quality and release gate checks.

---

## Phase Plan and Detailed Action Map

**Portfolio mapping (summary):**

| Phase | Primary portfolio outcome |
|-------|---------------------------|
| Phase 0 | **Good portfolio:** repo runs; eval artifact exists. |
| Phase 1 | **Hire-me core:** named benchmark(s), baselines, results table. |
| Phase 2 | **Hire-me depth:** robustness + quality metrics that support the “why.” |
| Phase 3 | **Trust:** CI + reproducibility = defensible numbers. |
| Phase 4 | **Optional:** only if needed to **match a specific SOTA claim**—not default. |

---

## Phase 0 - Stabilize Runtime Foundations (Weeks 1-2)

**Portfolio tier:** Good portfolio (prerequisite for any numeric claim).

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
- **Portfolio:** Document one command in README that produces a **saved metric artifact** (path named explicitly).

---

## Phase 1 - Evaluation and Benchmark Baseline (Weeks 3-6)

**Portfolio tier:** Hire-me immediately (primary); this phase is where the **resume line** is earned.

### Objective

Create reliable performance baselines across internal and external reasoning datasets—**optimized for one sharp headline benchmark**, not maximum adapter count.

### Actions

1) Add benchmark adapters with unified schema (**portfolio constraint:** ship adapters **only** for the benchmark(s) named in the README headline—typically **one** primary; add a second only after the main table is stable).
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
  - Treat unused adapter stubs as **out of scope** until the hire-me row is filled.

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
- **Hire-me:** README (or `docs/results/`) contains a **table**: metric name, baseline value, your value, Δ, and footnoted setup (model, decoding, split).
- **Hire-me:** “Why we win” paragraph uses the template in [Benchmark and SOTA Contract](#benchmark-and-sota-contract-defensible-comparisons)—no new features required, only honest alignment of claim and numbers.

---

## Phase 2 - Verification and Quality Controls (Weeks 7-10)

**Portfolio tier:** Hire-me depth—supports **trust** in the headline numbers (robustness, failure modes).

### Objective

Raise reasoning reliability with explicit consistency checks and quality metrics—**only in service of the benchmark story** (e.g. counterfactual validity if that is your thesis).

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
- **Portfolio:** At least one **ablation row** in the main results narrative (e.g. reward component off) tied to a **predicted** axis of improvement.

---

## Phase 3 - Operations and Reproducibility (Weeks 11-12)

**Portfolio tier:** Hire-me—**reproducibility is the difference** between “interesting repo” and “I believe these results.”

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
- **Hire-me:** Results table cites **config file path + commit** (or release tag) used to generate published numbers.

---

## Phase 4 - Advanced Reasoning Strategy Expansion (Weeks 13+)

**Portfolio tier:** **Optional / conditional.** Enter this phase only if a **specific SOTA or reviewer request** requires an extra decoding or search strategy **and** Phases 0–3 already support a headline table. Otherwise **stop at Phase 3** and write the technical note or arXiv instead.

### Objective

Move from monitoring-only strategy detection to active multi-strategy reasoning—**minimal additions**, each justified by a **pre-registered** ablation hypothesis.

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
- **Portfolio gate:** Phase 4 merges only with a **pre-written** one-line hypothesis and a **filled row** in the ablation table; otherwise defer.

---

## Prioritized Backlog (Execution Order)

**Portfolio-first ordering:** items that unlock **numbers and narrative** before items that expand **code surface**.

1. P0: Fix `train.py` runtime correctness and logger API usage.
2. P0: Fix `evaluate.py` logger API usage and reporting stability.
3. P0: Resolve theta typing compatibility in reward path.
4. P0: Add train/eval smoke integration tests.
5. P0 (**hire-me**): Draft README **results table shell** (metric names, TBD cells) and **baseline definition** so work stays measurement-driven.
6. P1: Refresh docs (`README.md`, `PROJECT_STRUCTURE.md`) to current architecture.
7. P1: Implement **one** primary benchmark adapter + eval path end-to-end (second benchmark only after first table is stable).
8. P1: Add reporting module and richer stratified outputs **as needed for the headline claim**.
9. P1: Write **short technical note** (`docs/technical-note.md`) outline: thesis, protocol, limitations—can later become arXiv.
10. P2: Add verification/metrics **only** if they appear in the public table or the “why” paragraph.
11. P2: CI workflow and runbook/repro governance.
12. P3: Phase 4 strategies / formal-method bridges—**only** with ablation justification; deprioritize domain expansion for resume impact.

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
  - **Portfolio:** One documented path from clone to **numeric artifact**.
- Phase 1:
  - **Primary benchmark:** 1 adapter + eval path complete; **secondary** optional.
  - README (or `docs/results/`) contains **≥2 numeric rows** (baseline + yours) for the same metric definition.
  - Per-dataset reporting completeness for **benchmarks you claim** in the headline: 100 percent.
- Phase 2:
  - Consistency violation detection precision/recall on synthetic fixtures: defined and tracked **if** cited publicly.
  - Calibration/hallucination metrics present in reports **only if** in the hire-me table or note.
- Phase 3:
  - CI pass rate and mean time to detect regressions tracked.
  - Reproducibility checks green for fixed seeds.
  - **Hire-me:** Published numbers traceable to **config + commit**.
- Phase 4:
  - Strategy ablation report generated per release candidate **if** Phase 4 is active.
  - Improvement over baseline metrics demonstrated for at least one strategy **with** a stated mechanism; otherwise skip Phase 4.

---

## Definition of Ready for Any New Feature

- Problem statement and expected value are documented.
- File-level design impact is identified.
- Test plan includes unit and at least one integration path.
- Acceptance criteria are explicit and measurable.
- Rollback path is defined for pipeline-impacting changes.
- **Portfolio:** For training/eval/reward changes, specify **which benchmark row** or **which sentence of the “why”** the change is meant to move; if none, defer.

---

## Suggested Immediate Next Actions (This Week)

1) Implement and merge Phase 0 fixes.
2) Add smoke tests and make them pass in CI (or local gate if CI not yet added).
3) Refresh docs to remove stale scaffold statements.
4) Create the **README results table skeleton** and **benchmark + baseline contract** (names, metrics, setup)—then scope Phase 1 adapters to that contract only.
5) Optionally: start `docs/technical-note.md` (or arXiv outline) in parallel so the story and the code stay aligned.

