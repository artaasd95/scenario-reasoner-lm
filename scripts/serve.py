#!/usr/bin/env python3
"""CLI entrypoint for scenario API server (SR-11)."""

from __future__ import annotations

import argparse
import logging
import signal
import sys
from pathlib import Path

import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Scenario Reasoner API server")
    parser.add_argument("--config", type=str, default="configs/serve.yaml")
    parser.add_argument("--host", type=str, default=None)
    parser.add_argument("--port", type=int, default=None)
    parser.add_argument("--provider", type=str, default=None, choices=["mock", "live"])
    return parser.parse_args()


def load_config(path: str) -> dict:
    cfg_path = Path(path)
    if not cfg_path.is_file():
        return {"host": "0.0.0.0", "port": 8000, "provider": "mock", "log_level": "info"}
    return yaml.safe_load(cfg_path.read_text(encoding="utf-8")) or {}


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    host = args.host or config.get("host", "0.0.0.0")
    port = args.port or config.get("port", 8000)
    provider = args.provider or config.get("provider", "mock")
    log_level = config.get("log_level", "info")

    logging.basicConfig(level=getattr(logging, log_level.upper(), logging.INFO))

    import uvicorn
    from src.serving.scenario_service import create_app

    app = create_app(provider_name=provider)
    server = uvicorn.Server(uvicorn.Config(app, host=host, port=port, log_level=log_level))

    def _shutdown(signum, frame):
        server.should_exit = True

    signal.signal(signal.SIGINT, _shutdown)
    signal.signal(signal.SIGTERM, _shutdown)
    server.run()


if __name__ == "__main__":
    main()
