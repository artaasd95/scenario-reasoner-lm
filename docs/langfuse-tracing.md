# Langfuse Trace Contract

**Vault seed:** S2-05 · Parent trace: `tenk_demo_run`

## Environment

See `.env.example` for required variables:

| Variable | Purpose |
| --- | --- |
| `LANGFUSE_PUBLIC_KEY` | Langfuse project public key |
| `LANGFUSE_SECRET_KEY` | Langfuse project secret key |
| `LANGFUSE_HOST` | Langfuse API host (default: cloud) |
| `ENTERPRISE_MODEL_NAME` | Model label attached to spans |
| `LLM_PROVIDER` | Provider id (`openai`, `azure`, `offline`) |
| `OPENAI_API_KEY` / provider keys | Live LLM calls when not offline |
| `DSPY_CACHE_DIR` | DSPy disk cache location |

When Langfuse keys are absent, `LangfuseTracer` records spans in-memory only (no-op export).

## Parent trace

| Field | Value |
| --- | --- |
| Name | `tenk_demo_run` |
| Input | `filing_id`, optional upload metadata |
| Metadata | `model_name`, `provider`, `dspy_cache_dir` |

## Child spans (ordered)

1. `loading` — read bundled or uploaded filing
2. `extraction` — DSPy evidence extraction
3. `chunking` — section parse + evidence chunks
4. `hypotheses` — missed-risk hypothesis generation
5. `scenario_build_1` … `scenario_build_5` — one span per scenario card
6. `critique` — grounding / plausibility critique
7. `ranking` — deduplicated ranking
8. `rendering` — UI / export artifact generation

## Span payload

Each span should record when available:

- **inputs** / **outputs** — stage-specific dicts
- **model parameters** — via trace metadata
- **evidence_chunk_ids** — list of chunk ids used
- **scores** — critique and ranking scores
- **latency_ms** — auto-computed on span end
- **retries** — retry count in metadata
- **failure** — error string when status is `error`

## Usage

```python
from src.tracing.trace_context import TenKDemoTrace, TraceSpanName

trace = TenKDemoTrace(filing_id="acme_corp_10k")
filing = trace.run_stage(
    TraceSpanName.LOADING,
    lambda: load_tenk_filing("acme_corp_10k"),
    inputs={"filing_id": "acme_corp_10k"},
)
trace.flush()
print(trace.langfuse_url)
```

Implementation: `src/tracing/langfuse_client.py`, `src/tracing/trace_context.py`.
