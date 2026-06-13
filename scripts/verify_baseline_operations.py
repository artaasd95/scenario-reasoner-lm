from __future__ import annotations

import json
from pathlib import Path

REQUIRED = [
    Path("benchmarks/baseline_frozen_qwen3_0.6b_run_001.json"),
    Path("benchmarks/baseline_qwen3_0.6b_artifacts/run_info.json"),
    Path("benchmarks/baseline_qwen3_0.6b_artifacts/metrics.json"),
]


def main() -> int:
    missing = [str(p) for p in REQUIRED if not p.exists()]
    if missing:
        print("Missing baseline artifacts:")
        for item in missing:
            print(f"- {item}")
        return 1

    for path in REQUIRED:
        json.loads(path.read_text(encoding="utf-8-sig"))

    print("Baseline operations verified.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
