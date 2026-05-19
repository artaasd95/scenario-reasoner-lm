# Enterprise Risk Eval Rubric (S4)

**Vault seeds:** S4-01 · S4-02 · S4-06 · **Milestone:** S4 — DSPy MIPRO and eval hardening

This document defines scoring for the tiny eval set (`data/eval/enterprise_risk_tiny.jsonl`)
and how to record baselines under `docs/eval/results/`.

> **Development note:** Do not run live eval or MIPRO jobs until API budget is reserved.
> Offline stubs and checked-in baseline artifacts support CI schema tests only.

## S2 evaluation criteria

| Criterion | What we measure | Pass threshold (tiny set) |
| --- | --- | ---: |
| **grounding** | Title aligns with golden keywords; evidence chunks present | ≥ 0.7 |
| **plausibility** | Causal chain length, critique score, missed-risk rationale | ≥ 0.7 |
| **severity_clarity** | Severity matches golden label for scenario index | ≥ 0.7 |
| **non_duplication** | Unique titles across the five-card set | ≥ 0.9 |
| **trace_completeness** | Per-card `trace_id` and parent run `trace_id` | ≥ 0.9 |

Implementation: `src/dspy_modules/eval_metrics.py` · Runner: `scripts/run_enterprise_eval.py`

## Pass / fail examples

### Grounding — pass

- Title contains golden keywords (e.g. `Taiwan`, `ASIC`) **and** `source_evidence` is non-empty.

### Grounding — fail

- Generic title (“Supply chain risk”) with no keyword overlap and no evidence chunks.

### Severity clarity — pass

- Scenario index 0 labeled `catastrophic` when golden `expected_severity` is `catastrophic`.

### Severity clarity — fail

- Same row labeled `medium` while filing narrative supports catastrophic OEM exposure.

### Non-duplication — pass

- Five distinct titles after ranking.

### Non-duplication — fail

- Two cards share the same normalized title (duplicate supply-chain framing).

## Known failure modes

| Tag | Description |
| --- | --- |
| `missing_evidence` | Card lacks `source_evidence` despite Risk Factors disclosure |
| `understated_cascade` | Chain stops before customer/regulatory impact |
| `covenant_blind_spot` | SEC/legal risk without debt covenant linkage |
| `staging_only` | Cyber scenario stops at staging, not production |
| `duplicate_supply_chain` | Overlaps Taiwan ASIC and Japan passive storylines |
| `vague_regulatory` | EU risk without revenue-at-risk implication |

Tags are listed per row in the JSONL (`failure_mode_tags`).

## BootstrapFewShot baseline (S4-01)

Recorded under:

- `docs/eval/results/bootstrap_fewshot/baseline_report.json`
- `docs/eval/results/bootstrap_fewshot/baseline_report.md`

Metadata fields: `model_id`, `optimizer`, `seed`, `run_id`, per-criterion aggregates.

Re-record when live budget is available:

```bash
python scripts/run_enterprise_eval.py --offline --optimizer BootstrapFewShot
```

## Optimizer choice (S4-03 / S4-06)

| When | Recommendation |
| --- | --- |
| Default / CI smoke | **BootstrapFewShot** (`ENABLE_MIPRO` unset) |
| After stable baseline gates | Enable **MIPRO** with `ENABLE_MIPRO=1` and equal budgets in `src/dspy_modules/optimize.py` |
| MIPRO import or compile failure | Automatic fallback to BootstrapFewShot (logged, non-fatal) |

**Rollback:** unset `ENABLE_MIPRO`, set `optimizer=BootstrapFewShot` in config/CLI, re-run eval against checked-in `docs/eval/baseline_scores.json`.

**Non-goals:** No headline quality claims without eval artifacts in `docs/eval/results/`.

## Before / after: BootstrapFewShot vs MIPRO (tiny eval)

_Placeholder until live MIPRO comparison is run. Values below are offline-stub scaffolding._

| Criterion | BootstrapFewShot | MIPRO (placeholder) | Δ |
| --- | ---: | ---: | ---: |
| grounding | 0.95 | 0.95 | 0.00 |
| plausibility | 0.85 | 0.86 | +0.01 |
| severity_clarity | 0.95 | 0.95 | 0.00 |
| non_duplication | 1.00 | 1.00 | 0.00 |
| trace_completeness | 1.00 | 1.00 | 0.00 |

Update this table from `docs/eval/results/comparison_report.json` after `scripts/compare_enterprise_optimizers.py` runs with budget.

## Decision log (optimizer adoption)

| Date | Decision |
| --- | --- |
| 2026-05-19 | Record BootstrapFewShot offline baseline before enabling MIPRO in CI |
| 2026-05-19 | MIPRO deferred in regression workflow until baseline gates stable (`workflow_dispatch` only) |
| 2026-05-19 | Adoption criteria: no S2 criterion regression vs `baseline_scores.json`; MIPRO optional until then |

## CI regression (S4-05)

Workflow template: `.github/workflows/enterprise_eval_regression.yml` (manual / disabled by default).

Checked-in gates: `docs/eval/baseline_scores.json`
