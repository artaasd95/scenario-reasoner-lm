"""VerificationReport builder for ScenarioArtifact (SR-21)."""

from __future__ import annotations

from typing import List

from src.serving.schemas import ClaimRecord, ClaimStatus, VerificationSummary
from src.verification.local_engine import verify_claim
from src.verification.numerical_verifier import extract_claims


def build_verification_report(texts: List[str]) -> VerificationSummary:
    claims_out: List[ClaimRecord] = []
    scores: List[float] = []

    for text in texts:
        for extracted in extract_claims(text):
            status = verify_claim(extracted)
            status_enum = ClaimStatus(status.value)
            claims_out.append(ClaimRecord(
                text=extracted.text,
                status=status_enum,
                detail=f"type={extracted.claim_type.value} value={extracted.value}",
            ))
            scores.append(1.0 if status_enum == ClaimStatus.VERIFIED else 0.5 if status_enum == ClaimStatus.FLAGGED else 0.0)

    overall = sum(scores) / len(scores) if scores else 0.0
    return VerificationSummary(overall_score=round(overall, 4), claims=claims_out)
