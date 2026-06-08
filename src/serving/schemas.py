"""Pydantic schemas for scenario generation API (SR-09)."""

from __future__ import annotations

from enum import Enum
from typing import Any, Dict, List, Literal, Optional
from uuid import uuid4

from pydantic import BaseModel, Field


class PathType(str, Enum):
    wide = "wide"
    bounded = "bounded"


class TailTag(str, Enum):
    none = "none"
    tail = "tail"
    non_possible = "non_possible"
    novel = "novel"


class PathRecord(BaseModel):
    path_id: str
    feasibility: float = Field(ge=0.0, le=1.0)
    tail_tag: TailTag = TailTag.none
    primary_quality_score: float = Field(ge=0.0, le=1.0, default=0.5)
    theta_stratum: str = ""
    text: str = ""


class Provenance(BaseModel):
    provider: str = "mock"
    model_id: str = ""
    datasource_id: str = ""
    generated_at: str = ""


class ClaimStatus(str, Enum):
    VERIFIED = "VERIFIED"
    FLAGGED = "FLAGGED"
    UNVERIFIED = "UNVERIFIED"


class ClaimRecord(BaseModel):
    text: str
    status: ClaimStatus
    detail: str = ""


class VerificationSummary(BaseModel):
    overall_score: float = Field(ge=0.0, le=1.0, default=0.0)
    claims: List[ClaimRecord] = Field(default_factory=list)


class ScenarioRequest(BaseModel):
    theta: Dict[str, Any]
    path_type: Literal["wide", "bounded"] = "bounded"
    n_paths: int = Field(ge=1, le=100, default=3)


class ScenarioArtifact(BaseModel):
    id: str = Field(default_factory=lambda: str(uuid4()))
    theta: Dict[str, Any]
    path_type: str
    n_paths: int
    paths: List[PathRecord]
    coherence: float = Field(ge=0.0, le=1.0)
    feasibility: float = Field(ge=0.0, le=1.0)
    tail_tags: List[str]
    provenance: Provenance
    verification_summary: VerificationSummary = Field(default_factory=VerificationSummary)
