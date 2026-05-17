"""
Enterprise risk scenario card schema.

Supports JSON export and test fixtures. Complements (does not replace)
causal scenario types in ``src.scenarios``.
"""

from __future__ import annotations

import json
import uuid
from dataclasses import asdict, dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional


class ScenarioSeverity(str, Enum):
    """Ordinal severity for enterprise risk scenarios."""

    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CATASTROPHIC = "catastrophic"


@dataclass
class EvidenceChunk:
    """
    Grounding unit extracted from a 10-K section.

    Attributes:
        section_name: SEC section label (e.g. ``Risk Factors``).
        chunk_id: Stable identifier within the filing run.
        source_span: Character or line span in the source document.
        quote_text: Verbatim excerpt used as evidence.
    """

    section_name: str
    chunk_id: str
    source_span: str
    quote_text: str

    def __post_init__(self) -> None:
        if not self.section_name.strip():
            raise ValueError("section_name must be non-empty")
        if not self.chunk_id.strip():
            raise ValueError("chunk_id must be non-empty")
        if not self.quote_text.strip():
            raise ValueError("quote_text must be non-empty")

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "EvidenceChunk":
        return cls(
            section_name=data["section_name"],
            chunk_id=data["chunk_id"],
            source_span=data.get("source_span", ""),
            quote_text=data["quote_text"],
        )


@dataclass
class EnterpriseRiskScenarioCard:
    """
    One catastrophic-but-plausible enterprise risk scenario grounded in 10-K evidence.

  All fields are required for a complete demo card except optional lists default empty.
    """

    title: str
    source_evidence: List[EvidenceChunk]
    causal_chain: List[str]
    missed_risk_rationale: str
    severity: str
    likelihood: str
    horizon: str
    confidence: float
    warning_signals: List[str] = field(default_factory=list)
    mitigations: List[str] = field(default_factory=list)
    trace_id: str = field(default_factory=lambda: str(uuid.uuid4()))

    VALID_SEVERITIES = tuple(s.value for s in ScenarioSeverity)
    VALID_LIKELIHOODS = ("low", "medium", "high")
    VALID_HORIZONS = ("0-6 months", "6-18 months", "18-36 months", "36+ months")

    def __post_init__(self) -> None:
        if not self.title.strip():
            raise ValueError("title must be non-empty")
        if not self.source_evidence:
            raise ValueError("source_evidence must contain at least one chunk")
        if len(self.causal_chain) < 2:
            raise ValueError("causal_chain must have at least two steps")
        if not self.missed_risk_rationale.strip():
            raise ValueError("missed_risk_rationale must be non-empty")
        if self.severity not in self.VALID_SEVERITIES:
            raise ValueError(
                f"severity must be one of {self.VALID_SEVERITIES}; got {self.severity!r}"
            )
        if self.likelihood not in self.VALID_LIKELIHOODS:
            raise ValueError(
                f"likelihood must be one of {self.VALID_LIKELIHOODS}; got {self.likelihood!r}"
            )
        if self.horizon not in self.VALID_HORIZONS:
            raise ValueError(
                f"horizon must be one of {self.VALID_HORIZONS}; got {self.horizon!r}"
            )
        if not (0.0 <= self.confidence <= 1.0):
            raise ValueError(f"confidence must be in [0, 1]; got {self.confidence}")

    def to_dict(self) -> Dict[str, Any]:
        return {
            "title": self.title,
            "source_evidence": [e.to_dict() for e in self.source_evidence],
            "causal_chain": list(self.causal_chain),
            "missed_risk_rationale": self.missed_risk_rationale,
            "severity": self.severity,
            "likelihood": self.likelihood,
            "horizon": self.horizon,
            "confidence": self.confidence,
            "warning_signals": list(self.warning_signals),
            "mitigations": list(self.mitigations),
            "trace_id": self.trace_id,
        }

    def to_json(self, indent: int = 2) -> str:
        return json.dumps(self.to_dict(), indent=indent)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "EnterpriseRiskScenarioCard":
        evidence = [
            EvidenceChunk.from_dict(e) if isinstance(e, dict) else e
            for e in data["source_evidence"]
        ]
        return cls(
            title=data["title"],
            source_evidence=evidence,
            causal_chain=list(data["causal_chain"]),
            missed_risk_rationale=data["missed_risk_rationale"],
            severity=data["severity"],
            likelihood=data["likelihood"],
            horizon=data["horizon"],
            confidence=float(data["confidence"]),
            warning_signals=list(data.get("warning_signals", [])),
            mitigations=list(data.get("mitigations", [])),
            trace_id=data.get("trace_id", str(uuid.uuid4())),
        )

    @classmethod
    def load_json_file(cls, path: str | Path) -> "EnterpriseRiskScenarioCard":
        with open(path, encoding="utf-8") as fh:
            return cls.from_dict(json.load(fh))


def scenario_card_from_dict(data: Dict[str, Any]) -> EnterpriseRiskScenarioCard:
    """Alias for :meth:`EnterpriseRiskScenarioCard.from_dict`."""
    return EnterpriseRiskScenarioCard.from_dict(data)
