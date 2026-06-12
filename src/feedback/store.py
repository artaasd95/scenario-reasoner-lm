"""JSONL-backed expert feedback store (SR-22)."""

from __future__ import annotations

import json
import sys
from contextlib import contextmanager
from pathlib import Path
from typing import Iterator, List, Optional
from uuid import uuid4

from src.feedback.expert_schema import ExpertAction, ExpertFeedback


@contextmanager
def _file_lock(path: Path) -> Iterator[None]:
    """Cross-platform exclusive lock for JSONL read-modify-write."""
    lock_path = path.with_suffix(path.suffix + ".lock")
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    with lock_path.open("w") as lock_file:
        if sys.platform == "win32":
            import msvcrt

            msvcrt.locking(lock_file.fileno(), msvcrt.LK_LOCK, 1)
        else:
            import fcntl

            fcntl.flock(lock_file.fileno(), fcntl.LOCK_EX)
        try:
            yield
        finally:
            if sys.platform == "win32":
                import msvcrt

                msvcrt.locking(lock_file.fileno(), msvcrt.LK_UNLCK, 1)
            else:
                import fcntl

                fcntl.flock(lock_file.fileno(), fcntl.LOCK_UN)


class FeedbackStore:
    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        if not self.path.is_file():
            self.path.touch()

    def _read_all(self) -> List[ExpertFeedback]:
        with _file_lock(self.path):
            rows: List[ExpertFeedback] = []
            for line in self.path.read_text(encoding="utf-8").splitlines():
                line = line.strip()
                if line:
                    rows.append(ExpertFeedback.from_dict(json.loads(line)))
            return rows

    def _write_all(self, rows: List[ExpertFeedback]) -> None:
        with _file_lock(self.path):
            with self.path.open("w", encoding="utf-8") as fh:
                for row in rows:
                    fh.write(json.dumps(row.to_dict()) + "\n")

    def list_recent(self, n: int = 5) -> List[ExpertFeedback]:
        """Return the most recent ``n`` feedback records."""
        rows = self._read_all()
        if n <= 0:
            return []
        return rows[-n:]

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
        with _file_lock(self.path):
            rows = self._read_all_unlocked()
            rows.append(fb)
            self._write_all_unlocked(rows)
        return fb

    def _read_all_unlocked(self) -> List[ExpertFeedback]:
        rows: List[ExpertFeedback] = []
        for line in self.path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if line:
                rows.append(ExpertFeedback.from_dict(json.loads(line)))
        return rows

    def _write_all_unlocked(self, rows: List[ExpertFeedback]) -> None:
        with self.path.open("w", encoding="utf-8") as fh:
            for row in rows:
                fh.write(json.dumps(row.to_dict()) + "\n")

    def get(self, feedback_id: str) -> Optional[ExpertFeedback]:
        with _file_lock(self.path):
            for line in self.path.read_text(encoding="utf-8").splitlines():
                line = line.strip()
                if not line:
                    continue
                row = ExpertFeedback.from_dict(json.loads(line))
                if row.feedback_id == feedback_id:
                    return row
        return None

    def update(self, feedback_id: str, **kwargs) -> Optional[ExpertFeedback]:
        with _file_lock(self.path):
            rows = self._read_all_unlocked()
            for i, row in enumerate(rows):
                if row.feedback_id == feedback_id:
                    data = row.to_dict()
                    data.update(kwargs)
                    updated = ExpertFeedback.from_dict(data)
                    rows[i] = updated
                    self._write_all_unlocked(rows)
                    return updated
        return None

    def list_by_status(self, status: str) -> List[ExpertFeedback]:
        return [r for r in self._read_all() if r.status == status]

    def delete(self, feedback_id: str) -> bool:
        with _file_lock(self.path):
            rows = self._read_all_unlocked()
            new_rows = [r for r in rows if r.feedback_id != feedback_id]
            if len(new_rows) == len(rows):
                return False
            self._write_all_unlocked(new_rows)
            return True
