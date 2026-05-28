# ADR: Financial Risk Analysis and Market-Making Reasoning

**Status:** Accepted (S6-01)  
**Date:** 2026-05-28

## Context

The showcase is already **enterprise financial risk** from 10-K filings. Contributors asked for an explicit **financial risk analysis** space and a **market making reasoning** sub-category that searches for strong reasoning approaches.

## Decision

1. **FinancialRiskTheta** (`src/scenarios/financial/financial_risk_theta.py`)
   - Extends enterprise filing-centric axes with `risk_lens` (credit, liquidity, operational, market), `stress_regime`, `valuation_horizon`
   - Output cards remain `EnterpriseRiskScenarioCard`-compatible where possible

2. **MarketMakingReasoningTheta**
   - Parameters: `spread_regime`, `inventory_pressure`, `reasoning_strategy_pool`, `search_budget`
   - **Purpose:** search over reasoning templates (e.g. inventory-skew narrative vs adverse-selection narrative), not execute trades
   - Maps to game-theoretic staged actions when joint fixtures are added (S6-04)

3. Headline **10-K demo unchanged**; financial θ is for extended scenarios and future fixtures.

## Consequences

- `financial_theta_to_enterprise_slice()` enables reuse of enterprise eval rubric subsets
- Market-making scenarios use `AlgorithmCard` + `SearchGraphMonitor` to compare reasoning paths

## Non-goals

| Non-goal | Rationale |
| --- | --- |
| Investment advice | Same as S2 |
| Live order books / exchange connectivity | Execution out of scope |
| Competing with enterprise demo for README headline | Financial extends, does not replace |

## References

- `src/scenarios/financial/`
- `docs/enterprise-risk-demo.md`
- `docs/scenario-search-extensions-contract.md`
