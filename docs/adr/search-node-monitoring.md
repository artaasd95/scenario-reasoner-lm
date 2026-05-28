# ADR: Search Graph Node Monitoring

**Status:** Accepted (S6-01)  
**Date:** 2026-05-28

## Context

Existing monitors (`CoTMonitor`, `ToTMonitor`, `AhaMonitor`) score **textual reasoning patterns**. S5 `reasoning_path_audit` scores **expected pipeline spans**. Neither records **per-node search statistics** (revisits, expansions, prunes) during tree or policy-guided search.

## Decision

Add `SearchGraphMonitor` in `src/search/graph_monitor.py`:

- `record_visit(node: NodeCard, algorithm_id: str)`
- `record_expansion`, `record_prune`
- `snapshot()` → `SearchGraphReport` with per-node aggregates and global branching factor estimate

Optional `span_id` on events for Langfuse correlation (no-op when keys absent).

S5 path fidelity and S6 node metrics are **complementary**: span order vs graph topology.

## Consequences

- Simulation runner (S6-02) can attach monitor to wide `path_mode` runs.
- Training can penalize excessive revisits or dead-end expansions via reward composer (future).

## Non-goals

- Replacing Langfuse; monitor works in-memory for tests
- Sub-millisecond profiling of every LLM token

## References

- `src/search/graph_monitor.py`
- `src/monitoring/reasoning_path_audit.py`
