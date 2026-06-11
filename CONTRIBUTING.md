# Contributing to scenario-reasoner-lm

## Setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install torch --index-url https://download.pytorch.org/whl/cpu
pip install -e ".[dev,serving]"
pytest tests/unit -m "not gpu and not integration"
```

## Branch policy

- Work on `develop`; keep `main` release-stable.
- Mock/offline paths remain the default for CI and local dev.

## BYOK

Runtime cloud LLM keys via environment variables only. Never commit API keys. Training and CI use platform-managed or mock providers.

## Tests

On-push CI runs ruff + CPU unit tests. GPU, integration, and eval smokes are `workflow_dispatch` only.
