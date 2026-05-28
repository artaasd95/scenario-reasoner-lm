# ADR: Algorithm Cards and Search Operators

**Status:** Accepted (S6-01)  
**Date:** 2026-05-28

## Context

Enterprise risk uses **scenario cards** for outputs. Search internals need the same auditability for **nodes**, **actions**, and **algorithms** so traces decompose as composable operators.

## Decision

Define three dataclasses in `src/search/cards.py`:

- `NodeCard` — vertex in the search graph
- `ActionCard` — decision taken at a node (includes vector slice and stage)
- `AlgorithmCard` — named operator with `SearchOperator` enum: `expand`, `rollout`, `backprop`, `prune`, `rank`, `select`

Algorithms emit cards; downstream code (monitors, Langfuse spans, JSONL logs) consumes them without parsing free-form CoT.

`AlgorithmCard.apply_to(node)` is a documentation hook only in S6-01 (returns metadata dict); real search policies wire in S6-02.

## Consequences

- Enterprise pipeline stages can be represented as `AlgorithmCard` sequences alongside existing `TraceSpanName` values.
- Preference builders and reward composers can optionally score operator sequences later.

## Non-goals

- A separate UI framework for editing cards
- Replacing DSPy modules with cards (cards wrap or annotate, not replace)

## References

- `src/search/cards.py`
- `src/tracing/trace_context.py`
