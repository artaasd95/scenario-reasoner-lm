# ADR: Game-Theoretic Action Space

**Status:** Accepted (S6-01)  
**Date:** 2026-05-28

## Context

Reasoning in this project is already modeled as search over trajectories (see `docs/scenario-search-formulation.md`). Game-theoretic scenarios make **A** explicit: at each stage t, a vector **u**_t ∈ ℝ^K (e.g. K=10) encodes preferences over discrete action types (expand, challenge opponent move, commit, backtrack, etc.).

## Decision

1. Introduce `GameTheoreticTheta` with:
   - `action_dim` (default 10)
   - `num_stages`
   - `action_vector`: flat list of length `action_dim * num_stages` or per-stage slices
   - `interaction_mode`: `single_agent` | `two_player_zero_sum` | `multi_agent` (stub enum)

2. Map vector → discrete action via `action_index_from_vector(slice, menu_size)` (argmax on softmax or top-1).

3. Optional `ActionManifold.project()` enforces box [0,1]^d or simplex constraints before discretization.

4. Game-theoretic θ **does not replace** `CausalTheta` or `EnterpriseRiskTheta`; use `game_theta_to_causal_slice()` for cross-harness reporting only.

## Consequences

- Simulation fixtures for `game` scenario type land in S6-02.
- Rewards and Ω can encode opponent feasibility without implementing full Nash computation in S6-01.

## Non-goals

- Equilibrium solvers, extensive-form game tree auto-generation from payoffs
- Real-time multi-agent APIs

## References

- `src/search/game_theta.py`
- `src/search/manifold.py`
- `docs/scenario-search-extensions-contract.md`
