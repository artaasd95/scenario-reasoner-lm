"""Feedback loop integration with synthetic coherence only (S10-07)."""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path

from src.data.schemas import FeedbackRecord

_REPO = Path(__file__).resolve().parents[2]


def _load_apply_feedback():
    spec = importlib.util.spec_from_file_location(
        "apply_feedback_mod",
        _REPO / "scripts" / "apply_feedback.py",
    )
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


class TestFeedbackLoop:
    def test_synthetic_coherence_pipeline(self, tmp_path):
        mod = _load_apply_feedback()
        load_feedback_jsonl = mod.load_feedback_jsonl
        apply_feedback = mod.apply_feedback

        incoming = tmp_path / "incoming"
        incoming.mkdir()
        records = [
            FeedbackRecord("r1", 0.9, {"chain_length": 2, "intervention_type": "direct", "domain": "physical", "difficulty": "easy"}),
            FeedbackRecord("r2", 0.3, {}),
        ]
        with (incoming / "batch.jsonl").open("w") as fh:
            for r in records:
                fh.write(json.dumps(r.to_dict()) + "\n")

        loaded = load_feedback_jsonl(incoming)
        result = apply_feedback(loaded, sink=str(tmp_path / "preferences"))
        assert result["feedback_count"] == 2
        assert result["preference_pairs"] == 1
