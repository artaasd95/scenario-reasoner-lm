# Scenario API Serving Reference (SR-14)

Mock-by-default HTTP API for scenario generation and evaluation.

## Endpoints

| Method | Path | Description |
|--------|------|-------------|
| `GET` | `/health` | Liveness check — returns `{"status": "ok"}` |
| `POST` | `/scenarios/generate` | Generate `ScenarioArtifact` from `ScenarioRequest` |
| `POST` | `/scenarios/evaluate` | Re-score an existing artifact |
| `GET` | `/scenarios/{id}` | Retrieve artifact by ID (in-memory dev store) |

## Quick start

```bash
pip install -e ".[serving]"
python scripts/serve.py --config configs/serve.yaml
```

## Example curl (mock provider)

```bash
curl -s http://localhost:8000/health

curl -s -X POST http://localhost:8000/scenarios/generate \
  -H "Content-Type: application/json" \
  -d '{"theta":{"chain_length":3,"domain":"physical"},"path_type":"bounded","n_paths":3}'
```

## Offline CLI

```bash
echo '{"chain_length":3,"domain":"physical"}' > theta.json
python scripts/generate.py --theta theta.json --output results/ --provider mock
```

## Provider switch

| Provider | Env | Notes |
|----------|-----|-------|
| `mock` (default) | none | Deterministic artifacts, no API keys |
| `live` | `ALLOW_LIVE_PROVIDER=1` | Gated live path (falls back to mock scaffold) |

Set in [`configs/serve.yaml`](../configs/serve.yaml) or `SCENARIO_PROVIDER=mock`.

## Schema

See [`src/serving/schemas.py`](../src/serving/schemas.py) — `ScenarioRequest`, `ScenarioArtifact`, `verification_summary`.
