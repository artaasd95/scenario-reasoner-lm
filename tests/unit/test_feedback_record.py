"""FeedbackRecord validation tests (S9-DP-07)."""

from __future__ import annotations

import json

from src.data.schemas import FeedbackRecord


class TestFeedbackRecord:
    def test_synthetic_coherence_to_record(self):
        rec = FeedbackRecord.from_dict(
            {
                "record_id": "fb-1",
                "coherence_score": 0.85,
                "theta": {
                    "chain_length": 2,
                    "intervention_type": "direct",
                    "domain": "physical",
                    "difficulty": "easy",
                },
                "path_quality": 0.9,
            }
        )
        assert rec.coherence_score >= 0.8
        d = rec.to_dict()
        assert json.loads(json.dumps(d))["record_id"] == "fb-1"
