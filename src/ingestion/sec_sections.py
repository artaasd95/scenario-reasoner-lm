"""
SEC 10-K section extraction from plain-text filings.

Extracts standard sections when present; missing sections are omitted.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Dict, List, Optional


@dataclass
class SECSection:
    """One extracted section from a 10-K."""

    name: str
    body: str
    start_offset: int
    end_offset: int

    @property
    def char_count(self) -> int:
        return len(self.body)


# Canonical section names mapped to heading patterns (order matters for greedy match).
_SECTION_PATTERNS: List[tuple[str, re.Pattern[str]]] = [
    ("Risk Factors", re.compile(r"(?im)^\s*item\s*1a\.?\s*[-–]?\s*risk\s+factors\s*$")),
    ("MD&A", re.compile(r"(?im)^\s*item\s*7\.?\s*[-–]?\s*management.+discussion\s*$")),
    ("Business", re.compile(r"(?im)^\s*item\s*1\.?\s*[-–]?\s*business\s*$")),
    ("Legal Proceedings", re.compile(r"(?im)^\s*item\s*3\.?\s*[-–]?\s*legal\s+proceedings\s*$")),
    ("Cybersecurity", re.compile(r"(?im)^\s*cybersecurity\s*$")),
    ("Regulatory", re.compile(r"(?im)^\s*regulatory\s+(?:exposure|matters|environment)\s*$")),
    ("Supply Chain", re.compile(r"(?im)^\s*supply\s+chain\s*$")),
]

# Fallback aliases in bundled sample (markdown-style headers)
_ALT_HEADERS: Dict[str, re.Pattern[str]] = {
    "Risk Factors": re.compile(r"(?im)^##\s*risk\s+factors\s*$"),
    "MD&A": re.compile(
        r"(?im)^##\s*management(?:'s|\s+)?\s*discussion\s+and\s+analysis\s*$"
    ),
    "Business": re.compile(r"(?im)^##\s*business\s+overview\s*$"),
    "Legal Proceedings": re.compile(r"(?im)^##\s*legal\s+proceedings\s*$"),
    "Cybersecurity": re.compile(r"(?im)^##\s*cybersecurity\s*$"),
    "Regulatory": re.compile(r"(?im)^##\s*regulatory\s+exposure\s*$"),
    "Supply Chain": re.compile(r"(?im)^##\s*supply\s+chain\s+dependencies\s*$"),
}


def _find_section_starts(text: str) -> List[tuple[int, str]]:
    hits: List[tuple[int, str]] = []
    for name, pattern in _SECTION_PATTERNS:
        m = pattern.search(text)
        if m:
            hits.append((m.start(), name))
    for name, pattern in _ALT_HEADERS.items():
        if any(n == name for _, n in hits):
            continue
        m = pattern.search(text)
        if m:
            hits.append((m.start(), name))
    hits.sort(key=lambda x: x[0])
    return hits


def extract_sections(
    text: str,
    section_names: Optional[List[str]] = None,
) -> Dict[str, SECSection]:
    """
    Extract SEC sections from filing text.

    Args:
        text: Full filing plain text.
        section_names: If provided, filter to these canonical names only.

    Returns:
        Dict mapping section name to :class:`SECSection`.
    """
    starts = _find_section_starts(text)
    if not starts:
        return {}

    sections: Dict[str, SECSection] = {}
    for i, (start, name) in enumerate(starts):
        end = starts[i + 1][0] if i + 1 < len(starts) else len(text)
        body = text[start:end].strip()
        if section_names and name not in section_names:
            continue
        sections[name] = SECSection(
            name=name,
            body=body,
            start_offset=start,
            end_offset=end,
        )
    return sections
