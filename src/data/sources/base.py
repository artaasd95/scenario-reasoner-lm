"""Base data source protocol (S9-DP-04)."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Dict, Iterator


class DataSource(ABC):
    @abstractmethod
    def iter_rows(self) -> Iterator[Dict[str, Any]]:
        ...
