"""Expert feedback store tests (SR-22)."""

from __future__ import annotations

from src.feedback.expert_schema import ExpertAction
from src.feedback.store import FeedbackStore


class TestFeedbackStore:
    def test_crud_and_list_by_status(self, tmp_path):
        store = FeedbackStore(tmp_path / "fb.jsonl")
        fb = store.create("scen-1", ExpertAction.APPROVE, note="ok", status="approved")
        assert store.get(fb.feedback_id) is not None
        assert len(store.list_by_status("approved")) == 1
        updated = store.update(fb.feedback_id, status="reviewed")
        assert updated is not None
        assert updated.status == "reviewed"
        assert store.delete(fb.feedback_id)
        assert store.get(fb.feedback_id) is None
