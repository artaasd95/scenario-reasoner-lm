# ADR: Simulation vs Measurement Paths

**Status:** Accepted (S5-01)  
**Vault seed:** S5-01  
**Date:** 2026-05-20

## Context

The repository serves two headline goals:

1. **Enterprise benchmark** — five source-grounded catastrophic scenarios from one bundled 10-K.
2. **Causal substrate** — template-generated θ sweeps for RLHF, robustness, and monitoring.

S5 introduces explicit **simulation** and **measurement** paths so wide exploratory search does not collide with scored eval harnesses.

## Decision

| Path | Definition | When to use |
| --- | --- | --- |
| **Simulation** | θ → world → trace | Generate or explore scenario families; audit reasoning spans; dry-run fixtures |
| **Measurement** | Scores, robustness slices, goal preservation | Regression gates, θ-stratified reports, optimizer comparison |

### Two simulation path types

1. **Wide / exploratory** — grids, tree expansion, Monte Carlo: small θ changes yield many paths (`path_mode=wide`).
2. **Bounded / staged** — good/bad/worst or fixed N stages (e.g. five enterprise cards; three causal stages) (`path_mode=bounded`).

### Θ mapping

`EnterpriseRiskTheta` remains source of truth for the 10-K demo. `enterprise_theta_to_causal_slice()` in `src/scenarios/theta_mapping.py` provides a causal parallel for cross-harness reporting only — causal RLHF code paths are not replaced.

## Consequences

- Runners default to **mock/smoke**; live calls require `ALLOW_LIVE_PROVIDER=1`.
- CI runs **unit/integration pytest only** until the full S5 sprint pipeline is scheduled.
- New domains and live SEC ingestion are **deferred** until paths are scaffolded.

## Non-goals (S5 decision log)

| Non-goal | Rationale |
| --- | --- |
| Live provider runs in dev/CI | Cost and flake; gated for approved sprint |
| New domains before scaffold | Keep θ taxonomies stable while runners land |
| Replacing causal Θ with enterprise Θ | Headline benchmark stays filing-centric |

## References

- `docs/scenario-simulation-paths.md`
- `src/scenarios/simulation_runner.py`
- `src/eval/scenario_measurement_schema.py`
