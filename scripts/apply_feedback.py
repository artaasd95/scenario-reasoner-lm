#!/usr/bin/env python3
"""Read feedback JSONL and emit preference pairs or policy weight updates (S10-WIRE-05)."""

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


def load_feedback_jsonl(path: Path) -> list[FeedbackRecord]:
    records: list[FeedbackRecord] = []
    if path.is_file():
        for line in path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if line:
                records.append(FeedbackRecord.from_dict(json.loads(line)))
        return records
    if not path.is_dir():
        return records
    for file in sorted(path.glob("*.jsonl")):
        records.extend(load_feedback_jsonl(file))
    return records


def apply_feedback_pairs(
    records: list[FeedbackRecord],
    *,
    coherence_threshold: float = 0.6,
) -> list[PreferencePair]:
    pairs: list[PreferencePair] = []
    for rec in records:
        if rec.coherence_score >= coherence_threshold:
            pairs.append(
                PreferencePair(
                    chosen=f"path_quality={rec.path_quality or rec.coherence_score}",
                    rejected="low_coherence_path",
                    prompt=rec.record_id,
                    theta=rec.theta,
                    metadata={"coherence_score": rec.coherence_score},
                )
            )
    return pairs


def apply_feedback_policy(
    records: list[FeedbackRecord],
    *,
    base_weights: dict[str, float] | None = None,
) -> dict[str, float]:
    weights = dict(base_weights or {"alpha_cot": 0.15, "beta_tot": 0.10, "gamma_aha": 0.05})
    if not records:
        return weights
    avg_coherence = sum(r.coherence_score for r in records) / len(records)
    delta = (avg_coherence - 0.5) * 0.1
    weights["alpha_cot"] = round(max(0.0, weights["alpha_cot"] + delta), 4)
    weights["beta_tot"] = round(max(0.0, weights["beta_tot"] + delta / 2), 4)
    return weights


def apply_feedback(
    records: list[FeedbackRecord],
    *,
    mode: str = "pairs",
    sink: str = "",
    coherence_threshold: float = 0.6,
    base_weights: dict[str, float] | None = None,
) -> dict:
    if mode == "policy":
        weights = apply_feedback_policy(records, base_weights=base_weights)
        return {
            "mode": "policy",
            "sink": sink,
            "feedback_count": len(records),
            "reward_weights": weights,
        }
    pairs = apply_feedback_pairs(records, coherence_threshold=coherence_threshold)
    return {
        "mode": "pairs",
        "sink": sink,
        "feedback_count": len(records),
        "preference_pairs": len(pairs),
        "pairs": [p.to_dict() for p in pairs],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Apply expert feedback to training artifacts")
    parser.add_argument("--config", type=str, default="configs/data/feedback.yaml")
    parser.add_argument("--incoming", type=str, default="feedback/incoming")
    parser.add_argument("--mode", choices=["pairs", "policy"], default="pairs")
    parser.add_argument("--coherence-threshold", type=float, default=0.6)
    args = parser.parse_args()

    config = load_pipeline_config(args.config)
    incoming = Path(args.incoming)
    records = load_feedback_jsonl(incoming)
    sink = config.metadata.get("preference_builder_sink", "data/processed/preferences")

    result = apply_feedback(
        records,
        mode=args.mode,
        sink=str(sink),
        coherence_threshold=args.coherence_threshold,
    )

    out_dir = Path(config.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    if args.mode == "pairs" and result.get("pairs"):
        pairs_path = out_dir / "preference_pairs.jsonl"
        with pairs_path.open("w", encoding="utf-8") as fh:
            for row in result["pairs"]:
                fh.write(json.dumps(row) + "\n")
        result["output"] = str(pairs_path)
    else:
        policy_path = out_dir / "policy_weights.json"
        policy_path.write_text(json.dumps(result.get("reward_weights", {}), indent=2), encoding="utf-8")
        result["output"] = str(policy_path)

    summary_path = out_dir / "feedback_apply.json"
    summary_path.write_text(json.dumps(result, indent=2), encoding="utf-8")
    logger.info("Wrote %s (%d records)", summary_path, len(records))
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s  %(message)s")
    main()
