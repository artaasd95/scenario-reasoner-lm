# Scenario Search Extensions Contract (S6)

**Vault seed:** S6-01 · **Milestone:** S6 — Game-theoretic search, algorithm cards, node monitoring, financial extensions  
**Status:** contract + stubs · **Depends on:** S2 (enterprise), S5 (simulation/measurement), causal substrate

## Demo goal (S6 does not replace S2)

S6 formalizes how **search algorithms** explore **structured decision spaces** inside the existing scenario formulation. The **headline benchmark remains unchanged:** five catastrophic enterprise scenarios from one bundled 10-K.

S6 adds:

1. **Game-theoretic action space** — staged action vectors embedded in Θ and constrained feasible sets (optional manifold projection).
2. **Algorithm cards** — composable operators: `NodeCard`, `ActionCard`, `AlgorithmCard` that algorithms use to traverse nodes and spaces.
3. **Search graph monitoring** — per-node visit/expansion/prune metrics during tree/MCTS/policy search (complements CoT/ToT/aha and S5 path audit).
4. **Financial risk analysis θ** — filing- and market-structure parameters for enterprise-style cards.
5. **Market-making reasoning θ** — search over reasoning templates/strategies (not live trading).

## Non-goals

| Non-goal | Rationale |
| --- | --- |
| Replace `ScenarioBase` or `EnterpriseRiskTheta` | Extend Θ; preserve causal RLHF and 10-K demo |
| Live trading or market making execution | Reasoning/search research only; no order placement |
| Full game-theory solver stack | Vectors + contracts first; equilibrium computation deferred |
| Learned manifold geometry in S6 | Box/simplex projection only; no training new embedding models in this milestone |
| Broad GRC / portfolio management product | Same as S2 enterprise non-goals |

## Formal alignment (S = X, Θ, T, A, R, Ω)

Game-theoretic scenarios specialize the action space **A**:

- **State** x_t: reasoning node (natural language or structured card id).
- **Action** a_t: index or mixture derived from `action_vector[stage]` (e.g. dim K=10 per stage).
- **Interaction:** multi-player extensions use `player_id` on `ActionCard` and joint θ; default is single-agent search with opponent modeled in Ω.
- **θ** includes `GameTheoreticTheta.action_vector`, `num_stages`, `action_dim`, `interaction_mode`.

Financial scenarios specialize **Θ** and **R**:

- `FinancialRiskTheta` — extends enterprise filing axes with `risk_lens`, `horizon`, `stress_regime`.
- `MarketMakingReasoningTheta` — `spread_regime`, `inventory_pressure`, `reasoning_strategy_pool` for template search.

## Algorithm cards (operator abstraction)

Three card types (see `src/search/cards.py`):

| Card | Role | Key fields |
| --- | --- | --- |
| `NodeCard` | Search tree vertex | `node_id`, `state_summary`, `depth`, `parent_id`, `theta_slice` |
| `ActionCard` | Decision at a node | `action_id`, `stage`, `vector_slice`, `label`, `player_id` |
| `AlgorithmCard` | Search policy / operator | `algorithm_id`, `operator` (expand, rollout, backprop, prune, rank), `config` |

Algorithms (MCTS, beam, greedy, enterprise pipeline stages) **emit** cards; monitors **consume** them. This mirrors `EnterpriseRiskScenarioCard` at the output layer but targets **search mechanics**.

## Search monitoring contract

`SearchGraphMonitor` records:

- `node_id`, `visit_count`, `expansion_count`, `prune_reason`, `algorithm_id`, `depth`
- Optional link to Langfuse span id (same no-op pattern as `reasoning_path_audit`)

S5 path audit checks **expected pipeline spans**; S6 node monitor checks **search topology** (branching, revisits, dead ends).

## Manifold (optional, constrained)

`ActionManifold` defines a feasible set F ⊂ ℝ^d (box or probability simplex). `project(action_vector)` returns a feasible vector before mapping to discrete actions. This is **not** a claim of learned geometric structure in S6—only **constraint enforcement** for mathematically well-defined θ.

## Phasing

| Phase | Deliverable |
| --- | --- |
| **S6-01 (now)** | This contract, ADRs, `src/search/*`, financial θ stubs, tests, `docs/project-track.md` |
| S6-02 | Simulation fixtures `game_bounded_default`, wire `SearchGraphMonitor` into simulation runner |
| S6-03 | Financial risk cards sharing enterprise schema; eval rubric slice |
| S6-04 | Market-making reasoning template search + measurement fixtures |

## Acceptance criteria (S6-01)

- [x] Contract states non-goals and north-star preservation.
- [x] `GameTheoreticTheta`, `FinancialRiskTheta`, `MarketMakingReasoningTheta` stubs with `to_dict()`.
- [x] Node / Action / Algorithm cards serialize round-trip.
- [x] `SearchGraphMonitor` unit-tested.
- [x] `theta_mapping.py` documents financial ↔ enterprise ↔ causal parallels.
- [x] `docs/project-track.md` orients contributors.

## Decision log

| Date | Decision | Rationale |
| --- | --- | --- |
| 2026-05-28 | **S6 extends search, not showcase** | 10-K demo remains headline; game/financial are parallel θ families |
| 2026-05-28 | **Action vector per stage** | Fixed dim (e.g. 10) maps to discrete action menu via argmax or sampling |
| 2026-05-28 | **Manifold = feasible set only** | Avoids scope creep into differential geometry training |
| 2026-05-28 | **Market making = reasoning search** | Fits “find good reasoning approaches”; avoids execution/regulatory scope |
| 2026-05-28 | **Cards as operators** | Same philosophy as scenario cards: auditable, JSON-serializable, composable |

## Related artifacts

| Area | Path |
| --- | --- |
| Master track | `docs/project-track.md` |
| ADR game-theoretic | `docs/adr/game-theoretic-action-space.md` |
| ADR cards | `docs/adr/algorithm-cards-operators.md` |
| ADR node monitor | `docs/adr/search-node-monitoring.md` |
| ADR financial | `docs/adr/financial-risk-and-market-making.md` |
| Implementation | `src/search/`, `src/scenarios/financial/` |
| Θ mapping | `src/scenarios/theta_mapping.py` |
