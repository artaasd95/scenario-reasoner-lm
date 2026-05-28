# Ideas and Plan — Scenario Reasoner LM

> **Orientation:** See [project-track.md](project-track.md) for the current layer cake and milestones.

## Vision (unchanged)

Build a training framework for open-source language models to reason into
structured scenarios using chain-of-thought and tree-of-thought prompting,
augmented by real-time monitoring of reasoning quality.

Formal view: scenarios are parameterized search problems (see
[scenario-search-formulation.md](scenario-search-formulation.md)).

## Current scope (implemented + S6 stubs)

| Area | Status |
| --- | --- |
| Causal / counterfactual scenarios | Implemented (`src/scenarios/causal/`) |
| Enterprise 10-K demo | Implemented (S2–S3) |
| Simulation vs measurement | Ongoing (S5) |
| DPO / rewards / monitors | Implemented |
| Game-theoretic action vectors | Contract + stubs (S6) |
| Algorithm cards (node / action / algorithm) | Implemented (`src/search/cards.py`) |
| Search graph node monitor | Implemented (`src/search/graph_monitor.py`) |
| Financial risk θ | Stub (`src/scenarios/financial/`) |
| Market-making reasoning search | Stub (not live trading) |

## Headline showcase (do not dilute)

Five source-grounded catastrophic enterprise scenarios from one bundled 10-K.
Financial and game-theoretic extensions **parallel** this path.

## Planned scenario types

- [x] Causal / counterfactual reasoning
- [ ] Game-theoretic interaction (S6-02 fixtures)
- [x] Enterprise / financial risk (10-K + `FinancialRiskTheta`)
- [ ] Market-making reasoning templates (S6-04)
- [ ] Multi-step arithmetic and algebra
- [ ] Legal and contract analysis
- [ ] Medical decision support
- [ ] Code debugging

## Training paradigms

- Supervised Fine-Tuning (SFT) on reasoning traces
- Reinforcement Learning from Human Feedback (RLHF) — DPO scaffold in repo
- Reward-based CoT / ToT selection
- Self-play and synthetic reasoning trace generation
- Optional: penalize poor search-graph topology via `SearchGraphMonitor` (future)

## Monitoring

- CoT / ToT / Aha monitors (textual patterns)
- S5 reasoning path audit (pipeline span fidelity)
- **S6** search graph monitor (per-node visits / expansions / prunes)

## Evaluation roadmap

- Exact match, F1 for final answers
- CoT faithfulness metrics
- Scenario-specific rubrics (enterprise eval in `docs/eval-enterprise-risk.md`)
- θ-stratified robustness (`robustness_eval.py`)
- Game / financial fixtures (S6-02+)
