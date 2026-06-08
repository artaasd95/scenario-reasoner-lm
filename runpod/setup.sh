#!/usr/bin/env bash
set -euo pipefail
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${REPO_ROOT}"
pip install -e ".[unsloth]" || pip install -e .
mkdir -p experiments/results/runpod experiments/adapters data/distilled
echo "[setup] done — python runpod/train.py"
