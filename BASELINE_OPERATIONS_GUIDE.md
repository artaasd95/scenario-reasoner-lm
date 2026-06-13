# Baseline Operations Guide

This repository uses a frozen local baseline bundle for experiment readiness.

## Baseline Files
- benchmarks/baseline_frozen_qwen3_0.6b_run_001.json
- benchmarks/baseline_qwen3_0.6b_artifacts/run_info.json
- benchmarks/baseline_qwen3_0.6b_artifacts/metrics.json

## Verify
python scripts/verify_baseline_operations.py

## Resume Semantics
- Baseline artifacts are immutable once frozen.
- New runs must not overwrite frozen baseline files.
