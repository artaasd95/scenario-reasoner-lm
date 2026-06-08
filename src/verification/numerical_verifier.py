"""Regex/heuristic numerical claim extractor (SR-19)."""

from __future__ import annotations

import re
from dataclasses import dataclass
from enum import Enum
from typing import List, Optional


class ClaimType(str, Enum):
    CVAR = "cvar"
    DRAWDOWN = "drawdown"
    LEVERAGE = "leverage"
    UNKNOWN = "unknown"


@dataclass
class ExtractedClaim:
    text: str
    claim_type: ClaimType
    value: Optional[float]
    unit: str = "%"


CVAR_RE = re.compile(r"CVaR\s+(\d+(?:\.\d+)?)\s*%", re.IGNORECASE)
DRAWDOWN_RE = re.compile(r"drawdown\s+(\d+(?:\.\d+)?)\s*%", re.IGNORECASE)
LEVERAGE_RE = re.compile(r"leverage\s+(?:ratio\s+)?(\d+(?:\.\d+)?)", re.IGNORECASE)


def extract_claims(text: str) -> List[ExtractedClaim]:
    claims: List[ExtractedClaim] = []
    for match in CVAR_RE.finditer(text):
        claims.append(ExtractedClaim(
            text=match.group(0),
            claim_type=ClaimType.CVAR,
            value=float(match.group(1)),
            unit="%",
        ))
    for match in DRAWDOWN_RE.finditer(text):
        claims.append(ExtractedClaim(
            text=match.group(0),
            claim_type=ClaimType.DRAWDOWN,
            value=float(match.group(1)),
            unit="%",
        ))
    for match in LEVERAGE_RE.finditer(text):
        claims.append(ExtractedClaim(
            text=match.group(0),
            claim_type=ClaimType.LEVERAGE,
            value=float(match.group(1)),
            unit="ratio",
        ))
    return claims
