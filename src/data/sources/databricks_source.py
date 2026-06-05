"""Databricks data source stub (S9-DP-04)."""

from __future__ import annotations

from typing import Any, Dict, Iterator, List, Optional

from src.data.sources.base import DataSource


class DatabricksDataSource(DataSource):
    """
    Stub adapter — production would use databricks-sql-connector.

    Accepts inline rows for tests and integration smoke paths.
    """

    def __init__(
        self,
        table: str = "",
        rows: Optional[List[Dict[str, Any]]] = None,
    ) -> None:
        self.table = table
        self._rows = rows or []

    def iter_rows(self) -> Iterator[Dict[str, Any]]:
        if not self._rows:
            raise NotImplementedError(
                "DatabricksDataSource is a stub; pass rows= for smoke tests "
                "or implement connector against table={self.table!r}"
            )
        yield from self._rows
