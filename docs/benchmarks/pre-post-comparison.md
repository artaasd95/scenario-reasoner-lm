# Pre vs Post Train Comparison

Compare a **base model** (pre-train) with its **LoRA adapter** (post-train) on the
same θ fixtures and seed.

## Command

```bash
export SCENARIO_MODELS_ROOT=/path/to/models

python scripts/compare_pre_post_train.py \
  --model-id qwen3-0.6b \
  --adapter-path $SCENARIO_TRAINED_MODELS_ROOT/qwen3-0.6b/unsloth_dpo_qwen3_0.6b \
  --fixtures data/eval/simulation_fixtures.json \
  --seed 42 \
  --training-method unsloth \
  --output docs/eval/results/pre_post/qwen3-0.6b_run01
```

## Outputs

| File | Content |
|------|---------|
| `pre_report.json` | Base model measurement report |
| `post_report.json` | Post-train adapter measurement report |
| `comparison_report.json` | Aggregate and θ-stratified deltas |
| `comparison_report.md` | Human-readable delta table |
| `eval_results.json` | HF model-card `eval_results` shape |

## Metrics

- **Coherence** — `on_target_composite` from goal-preservation fixtures
- **Feasibility** — `path_coverage` from simulation fixtures
- **Tail tags** — breakdown by `none` / `tail` / `non_possible` / `novel` (when present in pipeline rows)
- **θ-stratified** — per-fixture delta rows keyed by `fixture_id` and `theta`

## Integration test flow

1. Pre eval on fixtures
2. Mini DPO via `train.py` (or `--mini-train` hook)
3. Post eval with adapter path
4. Assert `comparison_report.md` contains delta columns

See `tests/integration/test_pre_post_compare.py`.
