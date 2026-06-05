"""SQL data source stub — expects pre-exported rows via query."""

from __future__ import annotations

import json
from typing import Any, Dict, Iterator, List, Optional

from src.data.sources.base import DataSource


class SQLDataSource(DataSource):
    """Load rows from a JSON export produced by an external SQL query."""

    def __init__(self, export_path: Optional[str] = None, rows: Optional[List[Dict[str, Any]]] = None) -> None:
        self.export_path = export_path
        self._rows = rows or []

    def iter_rows(self) -> Iterator[Dict[str, Any]]:
        if self._rows:
            yield from self._rows
            return
        if not self.export_path:
            return
        import pathlib

        text = pathlib.Path(self.export_path).read_text(encoding="utf-8")
        data = json.loads(text)
        for row in data if isinstance(data, list) else data.get("rows", []):
            yield row
