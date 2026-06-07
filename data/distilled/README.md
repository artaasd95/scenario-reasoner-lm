# Distilled Teacher Traces

Offline distillation dataset for scenario DPO training. Built in Colab via
[`notebooks/distill_teacher_colab.ipynb`](../../notebooks/distill_teacher_colab.ipynb).

## Layout

```
data/distilled/<name>/
  manifest.json       # schema_version, teacher, seed, fixture_refs, row_counts
  sft.jsonl           # CausalReasoningDataset rows
  dpo_pairs.jsonl     # PreferencePair rows
```

## manifest.json

```json
{
  "schema_version": "1.0.0",
  "name": "scenario_traces_v1",
  "teacher": "gpt-4o",
  "seed": 42,
  "fixture_refs": ["data/eval/simulation_fixtures.json"],
  "dpo_pairs_file": "dpo_pairs.jsonl",
  "sft_file": "sft.jsonl",
  "row_counts": {"dpo": 100, "sft": 100}
}
```

## dpo_pairs.jsonl (one object per line)

```json
{"prompt": "...", "chosen": "...", "rejected": "...", "theta": {}, "metadata": {}}
```

## Training

```yaml
training:
  data_source: distilled
  distilled_manifest: data/distilled/scenario_traces_v1/manifest.json
```

See [`docs/training.md`](../../docs/training.md) § Distillation.
