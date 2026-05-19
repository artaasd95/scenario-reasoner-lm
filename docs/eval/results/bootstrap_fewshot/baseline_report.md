# Enterprise Risk Eval Report

- **Schema:** 1.0.0
- **Created:** 2026-05-19T00:00:00+00:00
- **Model:** `offline-stub`
- **Optimizer:** `BootstrapFewShot`
- **Seed:** 42
- **Run ID:** `baseline-scaffold-s4-01`
- **Filing:** `acme_corp_10k`
- **Offline:** True
- **All pass:** True

- **Budget note:** BootstrapFewShot baseline: max_bootstrapped_demos=4, max_labeled_demos=8 (see optimize.py)

## Aggregate scores (S2 criteria)

| Criterion | Score | Threshold |
| --- | ---: | ---: |
| grounding | 0.95 | 0.7 |
| non_duplication | 1.0 | 0.9 |
| plausibility | 0.85 | 0.7 |
| severity_clarity | 0.95 | 0.7 |
| trace_completeness | 1.0 | 0.9 |

## Per-scenario

- **#0** `scaffold-0` — Single-foundry ASIC cutoff after Taiwan disruption — pass=True
- **#1** `scaffold-1` — SEC revenue recognition investigation triggers restatement spiral — pass=True
- **#2** `scaffold-2` — Staging breach escalates to production telemetry exposure — pass=True
- **#3** `scaffold-3` — EU machinery regulation blocks 19% revenue base — pass=True
- **#4** `scaffold-4` — Japanese sole-source passive shortage ends legacy platform support — pass=True

_Reserved for development: re-run `scripts/run_enterprise_eval.py` when API budget allows._
