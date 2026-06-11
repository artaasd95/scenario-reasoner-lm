# Custom LLM adapter (BYOK)

Use `CustomAdapter` when your organization hosts inference on an internal OpenAI-compatible endpoint.

1. Copy `configs/llm_custom.yaml` and set `base_url` + `api_key_env`.
2. Export the API key at runtime: `export MY_LLM_API_KEY=...` (never commit).
3. Optional: set `extra.fallback_provider: litellm` for cloud fallback.

Training jobs and CI must keep the default `mock` provider.
