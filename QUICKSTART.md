# Quick Start Guide: Enterprise Risk 10-K Demo

## Installation

```bash
# Clone and install
git clone <repo-url>
cd scenario-reasoner-lm

# Install dependencies
pip install -r requirements.txt
pip install -r requirements-enterprise.txt
```

## Running the Demo

### Option 1: Command Line (Recommended for CI/Testing)

```bash
# Run offline demo and output to artifacts
python scripts/run_enterprise_demo.py --offline --output artifacts/demo

# Results are in:
# - artifacts/demo/demo_result.json (complete pipeline output)
# - artifacts/demo/demo_result.json (trace ID for inspection)
```

### Option 2: Export to JSON & Markdown

```bash
# Generate scenarios and export to multiple formats
python scripts/export_demo_artifacts.py --offline --output artifacts/export

# Results are in:
# - artifacts/export/scenarios.json (structured data)
# - artifacts/export/scenarios.md (formatted for reading)
```

### Option 3: Interactive Streamlit UI

```bash
# Start the Streamlit app
streamlit run src/ui/streamlit_app.py

# Open browser to http://localhost:8501
# - Select bundled filing or upload your own 10-K
# - Click "Run demo"
# - View five scenario cards with evidence
# - Download JSON or Markdown
```

### Option 4: Docker

```bash
# Streamlit demo on http://localhost:8501
docker compose up enterprise-demo

# Or CLI in container
docker compose --profile cli up enterprise-demo-cli
```

## Output Format

### Scenario Card

Each card contains:
- **title** - Brief scenario name
- **severity** - `low`, `medium`, `high`, or `catastrophic`
- **likelihood** - `low`, `medium`, or `high`
- **horizon** - `0-6 months`, `6-18 months`, `18-36 months`, or `36+ months`
- **confidence** - 0.0 to 1.0 score
- **causal_chain** - Ordered steps leading to the outcome
- **missed_risk_rationale** - Why the 10-K understates this risk
- **source_evidence** - Grounded quotes from the filing
- **warning_signals** - Observable precursors
- **mitigations** - Possible countermeasures
- **trace_id** - Unique identifier for tracing

### Example Output

```json
{
  "title": "Taiwan ASIC Cutoff After Export Controls",
  "severity": "catastrophic",
  "likelihood": "medium",
  "horizon": "6-18 months",
  "confidence": 0.78,
  "causal_chain": [
    "Geopolitical tension increases",
    "Taiwan export controls tighten",
    "ASIC supply halts for 6+ months",
    "Production stops"
  ],
  "missed_risk_rationale": "10-K discloses sole-source but understates cascade to customer penalties.",
  "source_evidence": [
    {
      "section_name": "Risk Factors",
      "chunk_id": "acme_10k:Risk Factors:0",
      "quote_text": "Custom ASICs sourced from a single foundry partner in Taiwan..."
    }
  ],
  "warning_signals": [
    "Geopolitical news escalation",
    "Foundry allocation notices"
  ],
  "mitigations": [
    "Second-source ASIC qualification",
    "Supply buffer expansion"
  ],
  "trace_id": "550e8400-e29b-41d4-a716-446655440000"
}
```

## Testing

```bash
# Run all tests
python -m pytest tests/ -v

# Run specific test suite
python -m pytest tests/unit/test_ingestion.py -v      # Loader & sections
python -m pytest tests/unit/test_risk.py -v            # Schema & theta
python -m pytest tests/unit/test_dspy_modules.py -v    # Pipeline
python -m pytest tests/unit/test_tracing.py -v         # Tracing
python -m pytest tests/integration/ -v                  # End-to-end

# Run with coverage
python -m pytest tests/ --cov=src --cov-report=html
```

## Architecture

```
demo run
├── Load filing (tenk_loader.py)
├── Extract sections (sec_sections.py)
├── Chunk evidence (chunking.py)
├── Extract risks (extract_risks.py)
├── Generate hypotheses (generate_scenarios.py)
├── Build scenarios (generate_scenarios.py)
├── Critique scenarios (verify_scenarios.py)
├── Rank scenarios (verify_scenarios.py)
└── Export/render (streamlit_app.py, export_demo_artifacts.py)
    └── All stages traced to Langfuse (optional, graceful degradation)
```

## Bundled Sample Data

- **Filing:** `data/samples/tenk/acme_corp_10k.txt`
- **Company:** ACME Corporation
- **Fiscal Year:** 2025
- **Sections:** Risk Factors, MD&A, Cybersecurity, Regulatory, Supply Chain, Legal, Business

## Environment Variables (Optional)

For live LLM integration (not required for offline demo):

```bash
# Langfuse (optional, graceful no-op if absent)
LANGFUSE_PUBLIC_KEY=pk_...
LANGFUSE_SECRET_KEY=sk_...
LANGFUSE_HOST=https://cloud.langfuse.com

# LLM Provider
LLM_PROVIDER=offline              # Default: offline stubs
OPENAI_API_KEY=sk_...             # For OpenAI integration

# DSPy Cache
DSPY_CACHE_DIR=.dspy_cache

# Model Label
ENTERPRISE_MODEL_NAME=gpt-4       # Metadata in traces
```

See `.env.example` for all options.

## Performance

- **Offline mode:** ~2-5 seconds (no network I/O)
- **With Langfuse:** +1-2 seconds (trace export)
- **Memory:** ~200MB (Python + models)
- **CPU:** Single-threaded, all cores available

## Troubleshooting

| Issue | Solution |
|-------|----------|
| `ModuleNotFoundError: No module named 'dspy'` | `pip install -r requirements-enterprise.txt` |
| `FileNotFoundError: acme_corp_10k.txt` | Ensure running from repo root; bundled sample at `data/samples/tenk/` |
| Tests fail with import errors | Ensure `PYTHONPATH=/app` or run from repo root |
| Streamlit port 8501 already in use | `streamlit run src/ui/streamlit_app.py --server.port 8502` |
| Docker build fails | `docker compose build --no-cache` |

## Documentation

- **Demo Contract:** [docs/enterprise-risk-demo.md](../docs/enterprise-risk-demo.md)
- **Tracing Details:** [docs/langfuse-tracing.md](../docs/langfuse-tracing.md)
- **Implementation:** [IMPLEMENTATION_SUMMARY.md](../IMPLEMENTATION_SUMMARY.md)
- **Full README:** [README.md](../README.md)

## Key Concepts

- **Bundled Sample:** Complete offline 10-K in text format; no live SEC/EDGAR required
- **Offline Stubs:** DSPy modules include fallback implementations using pre-built scenarios
- **Graceful Degradation:** Tracing and LLM calls are optional; demo works without credentials
- **Scenario Card:** Structured representation of one risk scenario with evidence and reasoning
- **Trace ID:** Unique identifier connecting cards to execution traces (for auditing)
- **BootstrapFewShot:** Default DSPy optimizer; simpler than MIPRO for small eval sets

## Non-Goals

This demo is **not**:
- Financial advice or recommendation
- A replacement for traditional GRC platforms
- Dependent on training; causal RLHF is separate
- Using live SEC filings (bundled sample only)

---

**Questions?** See the full [IMPLEMENTATION_SUMMARY.md](../IMPLEMENTATION_SUMMARY.md) for architecture details and test coverage.
