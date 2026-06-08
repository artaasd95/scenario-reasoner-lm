"""Unit tests for apply_feedback (SR-07)."""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]


def _load_apply_feedback():
    spec = importlib.util.spec_from_file_location(
        "apply_feedback_mod",
        PROJECT_ROOT / "scripts" / "apply_feedback.py",
    )
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


_apply = _load_apply_feedback()
apply_feedback = _apply.apply_feedback
apply_feedback_pairs = _apply.apply_feedback_pairs
apply_feedback_policy = _apply.apply_feedback_policy
load_feedback_jsonl = _apply.load_feedback_jsonl

from src.data.schemas import FeedbackRecord


def _record(rid: str, score: float) -> FeedbackRecord:
    return FeedbackRecord(
        record_id=rid,
        coherence_score=score,
        theta={"chain_length": 3},
        path_quality=score,
        notes="",
    )


class TestApplyFeedback:
    def test_pairs_mode_high_coherence(self):
        records = [_record("a", 0.9), _record("b", 0.3), _record("c", 0.7)]
        result = apply_feedback(records, mode="pairs")
        assert result["preference_pairs"] == 2
        assert len(result["pairs"]) == 2

    def test_policy_mode_adjusts_weights(self):
        records = [_record("a", 0.9), _record("b", 0.8)]
        weights = apply_feedback_policy(records)
        assert weights["alpha_cot"] > 0.15

    def test_pairs_threshold(self):
        pairs = apply_feedback_pairs([_record("x", 0.5)], coherence_threshold=0.6)
        assert pairs == []

    def test_fixture_jsonl_roundtrip(self, tmp_path: Path):
        fixture = tmp_path / "feedback.jsonl"
        rows = [
            {"record_id": "r1", "coherence_score": 0.85, "theta": {"domain": "physical"}, "notes": ""},
            {"record_id": "r2", "coherence_score": 0.4, "theta": {}, "notes": "low"},
        ]
        fixture.write_text("\n".join(json.dumps(r) for r in rows), encoding="utf-8")
        loaded = load_feedback_jsonl(fixture)
        assert len(loaded) == 2
        result = apply_feedback(loaded, mode="pairs")
        assert result["preference_pairs"] == 1
