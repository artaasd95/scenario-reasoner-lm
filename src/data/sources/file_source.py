"""File JSONL data source."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Iterator

from src.data.sources.base import DataSource


class FileDataSource(DataSource):
    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)

    def iter_rows(self) -> Iterator[Dict[str, Any]]:
        with self.path.open(encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if line:
                    yield json.loads(line)
