"""JSONL-backed expert feedback store (SR-22)."""

from __future__ import annotations

import json
from pathlib import Path
from typing import List, Optional
from uuid import uuid4

from src.feedback.expert_schema import ExpertAction, ExpertFeedback


class FeedbackStore:
    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        if not self.path.is_file():
            self.path.touch()

    def _read_all(self) -> List[ExpertFeedback]:
        rows: List[ExpertFeedback] = []
        for line in self.path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if line:
                rows.append(ExpertFeedback.from_dict(json.loads(line)))
        return rows

    def _write_all(self, rows: List[ExpertFeedback]) -> None:
        with self.path.open("w", encoding="utf-8") as fh:
            for row in rows:
                fh.write(json.dumps(row.to_dict()) + "\n")

    def create(
        self,
        scenario_id: str,
        action: ExpertAction,
        *,
        note: str = "",
        theta: dict | None = None,
        status: str = "pending",
    ) -> ExpertFeedback:
        fb = ExpertFeedback(
            feedback_id=str(uuid4()),
            scenario_id=scenario_id,
            action=action,
            note=note,
            theta=theta or {},
            status=status,
        )
        rows = self._read_all()
        rows.append(fb)
        self._write_all(rows)
        return fb

    def get(self, feedback_id: str) -> Optional[ExpertFeedback]:
        for row in self._read_all():
            if row.feedback_id == feedback_id:
                return row
        return None

    def update(self, feedback_id: str, **kwargs) -> Optional[ExpertFeedback]:
        rows = self._read_all()
        for i, row in enumerate(rows):
            if row.feedback_id == feedback_id:
                data = row.to_dict()
                data.update(kwargs)
                updated = ExpertFeedback.from_dict(data)
                rows[i] = updated
                self._write_all(rows)
                return updated
        return None

    def list_by_status(self, status: str) -> List[ExpertFeedback]:
        return [r for r in self._read_all() if r.status == status]

    def delete(self, feedback_id: str) -> bool:
        rows = self._read_all()
        new_rows = [r for r in rows if r.feedback_id != feedback_id]
        if len(new_rows) == len(rows):
            return False
        self._write_all(new_rows)
        return True
