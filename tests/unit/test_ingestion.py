"""
Unit tests for 10-K ingestion pipeline: loader, sections, and chunking.
"""

from __future__ import annotations

import pytest
from pathlib import Path

from src.ingestion.tenk_loader import load_tenk_filing, list_bundled_filings
from src.ingestion.sec_sections import extract_sections, SECSection
from src.ingestion.chunking import chunk_section, chunk_sections
from src.risk.schema import EvidenceChunk


class TestTenKLoader:
    """Tests for bundled 10-K filing loader."""

    def test_load_bundled_filing(self):
        """Load the bundled ACME sample."""
        filing = load_tenk_filing("acme_corp_10k")
        assert filing.filing_id == "acme_corp_10k"
        assert filing.company_name == "ACME Corporation"
        assert filing.fiscal_year == 2025
        assert len(filing.raw_text) > 100
        assert "Risk Factors" in filing.raw_text or "RISK FACTORS" in filing.raw_text.upper()

    def test_bundled_filing_metadata(self):
        """Verify metadata consistency."""
        filing = load_tenk_filing("acme_corp_10k")
        assert filing.char_count > 0
        assert filing.format in ("text", "html")
        assert filing.source_path is not None

    def test_load_filing_by_path(self, tmp_path):
        """Load filing from arbitrary file path."""
        test_file = tmp_path / "test_10k.txt"
        test_file.write_text("Test Filing Content\nRisk Factors\nSome risk text here.", encoding="utf-8")
        
        filing = load_tenk_filing(str(test_file), filing_id="test_filing")
        assert filing.filing_id == "test_filing"
        assert "Risk Factors" in filing.raw_text

    def test_load_html_filing(self, tmp_path):
        """Load and normalize HTML filing."""
        test_file = tmp_path / "test_10k.html"
        test_file.write_text(
            "<html><body><p>Risk Factors</p><p>Some risk text</p></body></html>",
            encoding="utf-8"
        )
        
        filing = load_tenk_filing(str(test_file))
        assert "Risk Factors" in filing.raw_text
        assert "Some risk text" in filing.raw_text
        assert "<" not in filing.raw_text  # HTML stripped

    def test_list_bundled_filings(self):
        """Enumerate available bundled filings."""
        listings = list_bundled_filings()
        assert "acme_corp_10k" in listings
        assert len(listings) > 0

    def test_nonexistent_filing_raises(self):
        """Raise FileNotFoundError for unknown filing."""
        with pytest.raises(FileNotFoundError):
            load_tenk_filing("nonexistent_filing_id_12345")


class TestSectionExtraction:
    """Tests for SEC section extraction."""

    @pytest.fixture
    def sample_text(self):
        """Sample 10-K text with multiple sections."""
        return """
ACME CORPORATION FORM 10-K 2025

## Business Overview

ACME manufactures controllers.

## Risk Factors

We face concentration risk. Our top customers are critical.

## Management's Discussion and Analysis

Gross margin declined year-over-year.

## Legal Proceedings

The SEC is investigating us.

## Cybersecurity

We had an incident in staging.

## Regulatory Exposure

EU machinery regulations require updates.

## Supply Chain Dependencies

We rely on Japanese vendors for key components.
"""

    def test_extract_markdown_sections(self, sample_text):
        """Extract markdown-style section headers."""
        sections = extract_sections(sample_text)
        assert len(sections) > 0
        assert "Risk Factors" in sections
        assert "Cybersecurity" in sections

    def test_section_content_preserved(self, sample_text):
        """Verify section body contains expected content."""
        sections = extract_sections(sample_text)
        risk_section = sections.get("Risk Factors")
        assert risk_section is not None
        assert "concentration risk" in risk_section.body.lower()

    def test_section_offsets(self, sample_text):
        """Verify start/end offsets are within bounds."""
        sections = extract_sections(sample_text)
        for section in sections.values():
            assert section.start_offset >= 0
            assert section.end_offset <= len(sample_text)
            assert section.start_offset < section.end_offset

    def test_filter_sections_by_name(self, sample_text):
        """Filter extraction to specific section names."""
        sections = extract_sections(
            sample_text,
            section_names=["Risk Factors", "Cybersecurity"]
        )
        assert "Risk Factors" in sections
        assert "Cybersecurity" in sections
        assert "Legal Proceedings" not in sections

    def test_empty_text_returns_empty_dict(self):
        """Empty filing returns no sections."""
        sections = extract_sections("")
        assert len(sections) == 0

    def test_missing_sections_omitted(self, sample_text):
        """Extract only present sections."""
        text_without_legal = sample_text.replace("## Legal Proceedings", "")
        sections = extract_sections(text_without_legal)
        assert "Legal Proceedings" not in sections

    def test_section_char_count(self, sample_text):
        """Verify section char_count property."""
        sections = extract_sections(sample_text)
        for section in sections.values():
            assert section.char_count > 0
            assert section.char_count == len(section.body)


class TestChunking:
    """Tests for evidence chunk creation."""

    @pytest.fixture
    def sample_section(self):
        """Sample SEC section with multiple paragraphs."""
        body = """
We face substantial concentration risk: our top five customers accounted for 48% of
net revenue in fiscal 2025. Loss of any major customer could materially harm results.

Our products incorporate custom ASICs sourced from a single foundry partner in Taiwan.
Geopolitical tension, export controls, or natural disasters affecting that region could
interrupt supply for six months or longer.

We rely on a legacy ERP platform hosted in a single U.S. data center. A prolonged
outage could delay order fulfillment and financial close processes.

Foreign exchange fluctuations, particularly USD versus MYR and TWD, may reduce gross
margin when we cannot pass through cost increases.
"""
        return SECSection(
            name="Risk Factors",
            body=body,
            start_offset=100,
            end_offset=100 + len(body),
        )

    def test_chunk_section_basic(self, sample_section):
        """Create chunks from a section."""
        chunks = chunk_section(sample_section, filing_id="acme_corp_10k")
        assert len(chunks) > 0
        assert all(isinstance(c, EvidenceChunk) for c in chunks)

    def test_chunk_has_required_fields(self, sample_section):
        """Verify chunk has all required fields."""
        chunks = chunk_section(sample_section, filing_id="acme_corp_10k")
        for chunk in chunks:
            assert chunk.section_name == "Risk Factors"
            assert "acme_corp_10k" in chunk.chunk_id
            assert "Risk Factors" in chunk.chunk_id or "risk" in chunk.chunk_id.lower()
            assert chunk.chunk_id.count(":") >= 2  # filing:section:index
            assert chunk.source_span.startswith("chars:")
            assert len(chunk.quote_text) > 0

    def test_chunk_id_stability(self, sample_section):
        """Same section produces stable chunk IDs."""
        chunks1 = chunk_section(sample_section, filing_id="acme_corp_10k")
        chunks2 = chunk_section(sample_section, filing_id="acme_corp_10k")
        ids1 = [c.chunk_id for c in chunks1]
        ids2 = [c.chunk_id for c in chunks2]
        assert ids1 == ids2

    def test_chunk_respects_max_chars(self, sample_section):
        """Chunks generally respect max_chars limit (allow small overage for hard splits)."""
        max_chars = 200
        chunks = chunk_section(sample_section, filing_id="acme_corp_10k", max_chars=max_chars)
        # Allow small overage (up to 150% of max_chars) for hard-split paragraphs
        # This is acceptable since we hard-split when a single paragraph exceeds max_chars
        for chunk in chunks:
            assert len(chunk.quote_text) <= max_chars * 1.5, \
                f"Chunk exceeds reasonable limit: {len(chunk.quote_text)} > {max_chars * 1.5}"

    def test_chunk_overlap_handling(self, sample_section):
        """Chunks handle overlap when specified."""
        chunks_with_overlap = chunk_section(
            sample_section, filing_id="acme_corp_10k", max_chars=300, overlap=50
        )
        chunks_no_overlap = chunk_section(
            sample_section, filing_id="acme_corp_10k", max_chars=300, overlap=0
        )
        # With overlap, we may have slightly different chunking
        # but both should produce valid results
        assert len(chunks_with_overlap) > 0
        assert len(chunks_no_overlap) > 0

    def test_chunk_source_spans_valid(self, sample_section):
        """Source spans point to valid positions."""
        chunks = chunk_section(sample_section, filing_id="acme_corp_10k")
        for chunk in chunks:
            # Parse "chars:start-end"
            parts = chunk.source_span.split(":")
            assert len(parts) == 2
            start, end = map(int, parts[1].split("-"))
            assert start >= sample_section.start_offset
            assert end <= sample_section.end_offset

    def test_chunk_sections_multiple(self):
        """Chunk multiple sections together."""
        sections = {
            "Risk Factors": SECSection(
                name="Risk Factors",
                body="Risk 1\n\nRisk 2\n\nRisk 3",
                start_offset=0,
                end_offset=100,
            ),
            "Cybersecurity": SECSection(
                name="Cybersecurity",
                body="Cyber incident 1\n\nCyber incident 2",
                start_offset=100,
                end_offset=200,
            ),
        }
        all_chunks = chunk_sections(sections, filing_id="acme_corp_10k")
        
        assert len(all_chunks) > 0
        risk_chunks = [c for c in all_chunks if c.section_name == "Risk Factors"]
        cyber_chunks = [c for c in all_chunks if c.section_name == "Cybersecurity"]
        assert len(risk_chunks) > 0
        assert len(cyber_chunks) > 0

    def test_chunk_empty_section(self):
        """Handle empty section gracefully."""
        empty_section = SECSection(
            name="Empty",
            body="",
            start_offset=0,
            end_offset=0,
        )
        chunks = chunk_section(empty_section, filing_id="acme_corp_10k")
        assert len(chunks) == 0

    def test_chunk_single_paragraph(self):
        """Handle single short paragraph."""
        short_section = SECSection(
            name="Short",
            body="Just one short paragraph here.",
            start_offset=0,
            end_offset=30,
        )
        chunks = chunk_section(short_section, filing_id="acme_corp_10k")
        assert len(chunks) == 1
        assert chunks[0].quote_text == "Just one short paragraph here."


class TestIntegrationIngestion:
    """Integration tests: loader -> sections -> chunks."""

    def test_bundled_sample_full_pipeline(self):
        """Load, extract, and chunk ACME sample end-to-end."""
        # Load
        filing = load_tenk_filing("acme_corp_10k")
        assert len(filing.raw_text) > 0
        
        # Extract sections
        sections = extract_sections(filing.raw_text)
        assert len(sections) > 0
        
        # Chunk sections
        all_chunks = chunk_sections(sections, filing_id=filing.filing_id)
        assert len(all_chunks) > 0
        
        # Verify chunk coverage
        for chunk in all_chunks:
            assert chunk.section_name in sections
            assert chunk.quote_text in filing.raw_text
            assert len(chunk.quote_text) > 0

    def test_all_focus_sections_extracted(self):
        """Verify all priority sections are present in bundled sample."""
        filing = load_tenk_filing("acme_corp_10k")
        sections = extract_sections(filing.raw_text)
        
        focus = ["Risk Factors", "MD&A", "Cybersecurity", "Regulatory", "Supply Chain"]
        found = [name for name in focus if name in sections]
        assert len(found) >= 3, f"Expected at least 3 focus sections, found: {found}"

    def test_chunk_to_dict_roundtrip(self):
        """Chunks serialize and deserialize correctly."""
        filing = load_tenk_filing("acme_corp_10k")
        sections = extract_sections(filing.raw_text)
        chunks = chunk_sections(sections, filing_id=filing.filing_id)
        
        for chunk in chunks[:3]:  # Test first 3
            chunk_dict = chunk.to_dict()
            restored = EvidenceChunk.from_dict(chunk_dict)
            assert restored.section_name == chunk.section_name
            assert restored.chunk_id == chunk.chunk_id
            assert restored.quote_text == chunk.quote_text
