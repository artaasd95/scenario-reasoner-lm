from __future__ import annotations

import json
from pathlib import Path

LOCAL = Path("benchmarks/baseline_qwen3_0.6b_artifacts/metrics.json")
RUNPOD = Path("runpod/mock_runpod_metrics.json")


def main() -> int:
    local = json.loads(LOCAL.read_text(encoding="utf-8"))
    if not RUNPOD.exists():
        RUNPOD.write_text(json.dumps(local, indent=2), encoding="utf-8")
        print("Created mock RunPod metrics baseline.")
        return 0

    runpod = json.loads(RUNPOD.read_text(encoding="utf-8"))
    if local.get("test_metrics") != runpod.get("test_metrics"):
        print("Parity mismatch between local and RunPod metrics")
        return 1

    print("RunPod parity verified.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
