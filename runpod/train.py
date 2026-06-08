#!/usr/bin/env python3
"""RunPod-safe DPO training wrapper for scenario-reasoner-lm."""

from __future__ import annotations

import argparse
import glob
import os
import signal
import subprocess
import sys
import threading
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
STORAGE_DIR = REPO_ROOT / "storage"
DEFAULT_CONFIG = REPO_ROOT / "runpod" / "config.yaml"
DEFAULT_RUN_DIR = REPO_ROOT / "experiments" / "results" / "runpod"


class _Tee:
    def __init__(self, *streams):
        self._streams = streams

    def write(self, data: str) -> None:
        for stream in self._streams:
            stream.write(data)
            stream.flush()

    def flush(self) -> None:
        for stream in self._streams:
            stream.flush()


def _find_latest_checkpoint(output_dir: Path) -> Path | None:
    patterns = [
        str(output_dir / "checkpoint-*"),
        str(output_dir / "**" / "checkpoint-*"),
        str(output_dir / "dpo_checkpoint"),
    ]
    candidates: list[Path] = []
    for pattern in patterns:
        candidates.extend(Path(p) for p in glob.glob(pattern, recursive=True))
    if not candidates:
        return None
    return max(candidates, key=lambda p: p.stat().st_mtime)


def _disk_sync_loop(stop: threading.Event, interval_sec: int = 300) -> None:
    sys.path.insert(0, str(STORAGE_DIR))
    from disk_monitor import exceeds_threshold  # noqa: WPS433

    while not stop.wait(interval_sec):
        if exceeds_threshold(REPO_ROOT, 80.0):
            print("[runpod] disk >= 80% — triggering FTP sync")
            subprocess.run(
                [sys.executable, str(STORAGE_DIR / "ftp_sync.py"), "--check-threshold", "80"],
                cwd=REPO_ROOT,
                check=False,
            )


def main() -> int:
    parser = argparse.ArgumentParser(description="RunPod DPO training wrapper")
    parser.add_argument("--config", default=str(DEFAULT_CONFIG))
    parser.add_argument("--resume", default="auto")
    parser.add_argument("--output-dir", default=str(DEFAULT_RUN_DIR))
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    log_path = output_dir / "train.log"

    env = {**os.environ, "RUNPOD_OUTPUT_DIR": str(output_dir)}
    if args.resume == "auto":
        latest = _find_latest_checkpoint(output_dir)
        if latest is not None:
            env["RESUME_FROM_CHECKPOINT"] = str(latest)
            print(f"[runpod] auto-resume from {latest}")
    elif args.resume not in ("none", ""):
        env["RESUME_FROM_CHECKPOINT"] = args.resume

    cmd = [
        sys.executable,
        str(REPO_ROOT / "scripts" / "train.py"),
        "--config",
        args.config,
        "--output-dir",
        str(output_dir),
    ]

    stop_event = threading.Event()
    monitor = threading.Thread(target=_disk_sync_loop, args=(stop_event,), daemon=True)
    monitor.start()

    proc: subprocess.Popen[str] | None = None

    def _handle_signal(signum: int, _frame: object) -> None:
        print(f"[runpod] received signal {signum}")
        if proc is not None and proc.poll() is None:
            proc.send_signal(signum)

    signal.signal(signal.SIGTERM, _handle_signal)
    signal.signal(signal.SIGINT, _handle_signal)

    with log_path.open("a", encoding="utf-8") as log_file:
        log_file.write(f"\n--- runpod train start {time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())} ---\n")
        tee = _Tee(sys.stdout, log_file)
        proc = subprocess.Popen(
            cmd,
            cwd=REPO_ROOT,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            env=env,
        )
        assert proc.stdout is not None
        for line in proc.stdout:
            tee.write(line)
        rc = proc.wait()

    stop_event.set()
    monitor.join(timeout=1)
    return rc


if __name__ == "__main__":
    raise SystemExit(main())
