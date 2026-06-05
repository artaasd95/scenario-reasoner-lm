#!/usr/bin/env python3
"""
Validate feedback YAML and stub apply path (S9-DP-07 / S10-07).

Reads coherence scores from feedback/incoming/, emits FeedbackRecord rows.
Full preference_builder sink logic is stubbed for S10-07 integration.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.data.pipeline.config import load_pipeline_config
from src.data.schemas import FeedbackRecord, PreferencePair

logger = logging.getLogger(__name__)


def load_feedback_incoming(incoming_dir: Path) -> list[FeedbackRecord]:
    records: list[FeedbackRecord] = []
    if not incoming_dir.is_dir():
        return records
    for path in sorted(incoming_dir.glob("*.jsonl")):
        for line in path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line:
                continue
            records.append(FeedbackRecord.from_dict(json.loads(line)))
    return records


def apply_feedback_stub(
    records: list[FeedbackRecord],
    *,
    sink: str,
) -> dict:
    """
    Stub: document preference_builder sink; build PreferencePair when coherence high.
    """
    pairs: list[PreferencePair] = []
    for rec in records:
        if rec.coherence_score >= 0.6:
            pairs.append(
                PreferencePair(
                    chosen=f"path_quality={rec.path_quality or rec.coherence_score}",
                    rejected="low_coherence_path",
                    prompt=rec.record_id,
                    theta=rec.theta,
                    metadata={"coherence_score": rec.coherence_score},
                )
            )
    return {
        "sink": sink,
        "feedback_count": len(records),
        "preference_pairs": len(pairs),
        "pairs": [p.to_dict() for p in pairs],
        "note": "Full preference_builder integration in S10-07",
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Apply feedback stub")
    parser.add_argument(
        "--config",
        type=str,
        default="configs/data/feedback.yaml",
        help="Feedback YAML config",
    )
    parser.add_argument("--incoming", type=str, default="feedback/incoming")
    args = parser.parse_args()

    config = load_pipeline_config(args.config)
    incoming = Path(args.incoming)
    records = load_feedback_incoming(incoming)
    sink = config.metadata.get("preference_builder_sink", "data/processed/preferences")

    result = apply_feedback_stub(records, sink=str(sink))
    out_dir = Path(config.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "feedback_apply_stub.json"
    out_path.write_text(json.dumps(result, indent=2), encoding="utf-8")
    logger.info("Wrote %s (%d records)", out_path, len(records))
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s  %(message)s")
    main()
