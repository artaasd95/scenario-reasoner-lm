# Deployment Guide (SR-18)

## Docker (scenario API)

```bash
docker build -f Dockerfile.serve -t scenario-reasoner-api .
docker run --rm -p 8000:8000 -e SCENARIO_PROVIDER=mock scenario-reasoner-api
curl -s http://localhost:8000/health
```

## Docker Compose

```bash
# Milestone exit: mock API on :8000
docker compose up scenario-api

# Optional enterprise Streamlit demo
docker compose --profile demo up enterprise-demo

# Optional expert review UI
docker compose --profile review up review-ui
```

## Configuration

| Key | File | Default | Description |
|-----|------|---------|-------------|
| `provider` | `configs/serve.yaml` | `mock` | `mock` or `live` |
| `host` | `configs/serve.yaml` | `0.0.0.0` | Bind address |
| `port` | `configs/serve.yaml` | `8000` | HTTP port |
| `SCENARIO_PROVIDER` | env | `mock` | Override provider in compose |
| `ALLOW_LIVE_PROVIDER` | env | `0` | Must be `1` for live provider |

## Local overrides

[`docker-compose.override.yml`](../docker-compose.override.yml) mounts `configs/` and `data/` for local dev.

## Troubleshooting

| Issue | Fix |
|-------|-----|
| `/health` fails | Wait for `start_period` (15s); check logs: `docker compose logs scenario-api` |
| Port 8000 in use | Change mapping: `"8001:8000"` in compose |
| Live provider 403 | Set `ALLOW_LIVE_PROVIDER=1` |
| Import errors in container | Rebuild: `docker compose build --no-cache scenario-api` |

See also [serving.md](serving.md) for API reference.
