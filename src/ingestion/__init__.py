"""10-K filing ingestion, section extraction, and evidence chunking."""

from src.ingestion.chunking import chunk_sections, evidence_chunks_from_sections
from src.ingestion.sec_sections import SECSection, extract_sections
from src.ingestion.tenk_loader import TenKFiling, load_tenk_filing

__all__ = [
    "TenKFiling",
    "load_tenk_filing",
    "SECSection",
    "extract_sections",
    "chunk_sections",
    "evidence_chunks_from_sections",
]
