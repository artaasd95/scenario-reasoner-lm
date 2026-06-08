#!/usr/bin/env python3
"""Offline scenario artifact generator CLI (SR-12)."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.serving.providers import get_provider
from src.serving.schemas import ScenarioRequest


def main() -> int:
    parser = argparse.ArgumentParser(description="Generate ScenarioArtifact offline")
    parser.add_argument("--theta", type=str, required=True, help="Path to theta JSON file")
    parser.add_argument("--output", type=str, required=True, help="Output directory")
    parser.add_argument("--provider", type=str, default="mock", choices=["mock", "live"])
    parser.add_argument("--path-type", type=str, default="bounded", choices=["wide", "bounded"])
    parser.add_argument("--n-paths", type=int, default=3)
    args = parser.parse_args()

    theta = json.loads(Path(args.theta).read_text(encoding="utf-8"))
    request = ScenarioRequest(theta=theta, path_type=args.path_type, n_paths=args.n_paths)
    provider = get_provider(args.provider)
    artifact = provider.generate(request)

    out_dir = Path(args.output)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "scenario_artifact.json"
    out_path.write_text(artifact.model_dump_json(indent=2), encoding="utf-8")
    print(f"Wrote {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
