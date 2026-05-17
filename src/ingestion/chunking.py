"""
Evidence chunking for 10-K sections.

Each chunk carries section name, chunk id, source span, and quote text.
"""

from __future__ import annotations

import hashlib
import re
from typing import Dict, List, Optional

from src.ingestion.sec_sections import SECSection
from src.risk.schema import EvidenceChunk

_DEFAULT_MAX_CHARS = 600
_DEFAULT_OVERLAP = 80


def _split_paragraphs(body: str) -> List[str]:
    parts = re.split(r"\n\s*\n", body)
    return [p.strip() for p in parts if p.strip()]


def chunk_section(
    section: SECSection,
    filing_id: str,
    max_chars: int = _DEFAULT_MAX_CHARS,
    overlap: int = _DEFAULT_OVERLAP,
) -> List[EvidenceChunk]:
    """Split one section into overlapping evidence chunks."""
    chunks: List[EvidenceChunk] = []
    paragraphs = _split_paragraphs(section.body)
    if not paragraphs:
        return chunks

    buffer = ""
    buffer_start = section.start_offset
    chunk_index = 0

    for para in paragraphs:
        candidate = f"{buffer}\n\n{para}".strip() if buffer else para
        if len(candidate) <= max_chars:
            buffer = candidate
            continue

        if buffer:
            chunk_id = f"{filing_id}:{section.name}:{chunk_index}"
            span_end = buffer_start + len(buffer)
            chunks.append(
                EvidenceChunk(
                    section_name=section.name,
                    chunk_id=chunk_id,
                    source_span=f"chars:{buffer_start}-{span_end}",
                    quote_text=buffer,
                )
            )
            chunk_index += 1
            tail = buffer[-overlap:] if overlap and len(buffer) > overlap else ""
            buffer_start = span_end - len(tail)
            buffer = f"{tail}\n\n{para}".strip() if tail else para
        else:
            # Single paragraph longer than max_chars — hard split
            for offset in range(0, len(para), max_chars - overlap):
                piece = para[offset : offset + max_chars]
                chunk_id = f"{filing_id}:{section.name}:{chunk_index}"
                abs_start = section.start_offset + offset
                chunks.append(
                    EvidenceChunk(
                        section_name=section.name,
                        chunk_id=chunk_id,
                        source_span=f"chars:{abs_start}-{abs_start + len(piece)}",
                        quote_text=piece,
                    )
                )
                chunk_index += 1
            buffer = ""
            buffer_start = section.end_offset

    if buffer:
        chunk_id = f"{filing_id}:{section.name}:{chunk_index}"
        span_end = buffer_start + len(buffer)
        chunks.append(
            EvidenceChunk(
                section_name=section.name,
                chunk_id=chunk_id,
                source_span=f"chars:{buffer_start}-{span_end}",
                quote_text=buffer,
            )
        )

    return chunks


def chunk_sections(
    sections: Dict[str, SECSection],
    filing_id: str,
    max_chars: int = _DEFAULT_MAX_CHARS,
    overlap: int = _DEFAULT_OVERLAP,
) -> List[EvidenceChunk]:
    """Chunk all sections and return a flat list."""
    all_chunks: List[EvidenceChunk] = []
    for section in sections.values():
        all_chunks.extend(
            chunk_section(section, filing_id, max_chars=max_chars, overlap=overlap)
        )
    return all_chunks


def evidence_chunks_from_sections(
    sections: Dict[str, SECSection],
    filing_id: str,
    focus_sections: Optional[List[str]] = None,
) -> List[EvidenceChunk]:
    """Chunk only focus sections when specified."""
    if focus_sections:
        filtered = {k: v for k, v in sections.items() if k in focus_sections}
        return chunk_sections(filtered, filing_id)
    return chunk_sections(sections, filing_id)


def stable_chunk_id(filing_id: str, section_name: str, quote_text: str) -> str:
    """Deterministic chunk id from content hash (for fixtures)."""
    digest = hashlib.sha256(quote_text.encode("utf-8")).hexdigest()[:12]
    slug = re.sub(r"[^a-z0-9]+", "_", section_name.lower()).strip("_")
    return f"{filing_id}:{slug}:{digest}"
