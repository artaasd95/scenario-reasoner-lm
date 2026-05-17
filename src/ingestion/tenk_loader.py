"""
10-K filing loader.

Supports bundled plain-text or HTML filings before live SEC/EDGAR ingestion.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Union

# Repo root: src/ingestion/tenk_loader.py -> parents[2]
_REPO_ROOT = Path(__file__).resolve().parents[2]
_DEFAULT_SAMPLE = _REPO_ROOT / "data" / "samples" / "tenk" / "acme_corp_10k.txt"
_BUNDLED_FILINGS = {
    "acme_corp_10k": _DEFAULT_SAMPLE,
}


@dataclass
class TenKFiling:
    """Loaded 10-K document."""

    filing_id: str
    company_name: str
    fiscal_year: int
    raw_text: str
    source_path: Optional[str] = None
    format: str = "text"

    @property
    def char_count(self) -> int:
        return len(self.raw_text)


def _strip_html(html: str) -> str:
    text = re.sub(r"<(script|style)[^>]*>.*?</\1>", " ", html, flags=re.I | re.S)
    text = re.sub(r"<[^>]+>", " ", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def load_tenk_filing(
    source: Union[str, Path],
    filing_id: Optional[str] = None,
    company_name: str = "ACME Corporation",
    fiscal_year: int = 2025,
) -> TenKFiling:
    """
    Load a 10-K from a bundled id, file path, or raw string path.

    Args:
        source: Bundled ``filing_id`` (e.g. ``acme_corp_10k``) or filesystem path.
        filing_id: Override filing identifier (defaults from source).
        company_name: Display name for metadata.
        fiscal_year: Fiscal year label.

    Returns:
        :class:`TenKFiling` with normalized plain text.
    """
    source_str = str(source)
    path: Optional[Path] = None
    resolved_id = filing_id or source_str

    if source_str in _BUNDLED_FILINGS:
        path = _BUNDLED_FILINGS[source_str]
        resolved_id = source_str
    else:
        candidate = Path(source_str)
        if candidate.is_file():
            path = candidate
            resolved_id = filing_id or candidate.stem

    if path is not None:
        raw = path.read_text(encoding="utf-8")
        fmt = "html" if path.suffix.lower() in {".htm", ".html"} else "text"
        if fmt == "html":
            raw = _strip_html(raw)
        return TenKFiling(
            filing_id=resolved_id,
            company_name=company_name,
            fiscal_year=fiscal_year,
            raw_text=raw,
            source_path=str(path),
            format=fmt,
        )

    raise FileNotFoundError(
        f"Unknown bundled filing {source_str!r} and path does not exist: {source_str}"
    )


def list_bundled_filings() -> list[str]:
    return list(_BUNDLED_FILINGS.keys())
