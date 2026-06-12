# Context management

Scenario Reasoner LM uses a **token-budget context manager** in `src.llm_integration.context` to prevent 10-K filing excerpts and runtime prompts from exceeding model windows.

## Method

1. **Model registry** — `configs/model_context.yaml` defines per-model limits.
2. **Token estimation** — `tiktoken` when available, else chars/4 heuristic.
3. **Priority packing** — `ContextBudget.assemble()` keeps higher-priority segments first.
4. **10-K excerpt** — `src/dspy_modules/filing_context.py` ranks chunks by section relevance (Risk Factors → MD&A → …) and packs them into `build_filing_excerpt()` used by `ExtractRisksModule`.

## Settings

| Variable | Purpose |
|----------|---------|
| `SCENARIO_MAX_CONTEXT_TOKENS` | Cap input context for filing excerpts and runtime calls |
| `SCENARIO_MODEL_CONTEXT_PATH` | Override registry YAML path |
| `ENTERPRISE_MODEL_NAME` | Model id used when resolving excerpt budgets |

## Why section-ranked packing

Concatenating all 600-char chunks can exceed cloud model limits on long 10-K filings. Ranking by SEC section relevance keeps catastrophic-risk evidence in context while dropping lower-value sections first.
