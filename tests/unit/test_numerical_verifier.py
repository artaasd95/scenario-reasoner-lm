"""Numerical verifier tests (SR-19)."""

from __future__ import annotations

import pytest

from src.verification.local_engine import VerifyStatus, verify_claim
from src.verification.numerical_verifier import ClaimType, extract_claims


FIXTURES = [
    ("CVaR 12.3% expected tail loss", ClaimType.CVAR, 12.3),
    ("drawdown 8.7% from peak", ClaimType.DRAWDOWN, 8.7),
    ("leverage ratio 2.5 on book", ClaimType.LEVERAGE, 2.5),
    ("CVaR 5.0% and drawdown 3.2%", ClaimType.CVAR, 5.0),
    ("no numbers here", None, None),
    ("CVaR 99.9% extreme", ClaimType.CVAR, 99.9),
    ("drawdown 0.5% minor", ClaimType.DRAWDOWN, 0.5),
    ("leverage 10.0 high", ClaimType.LEVERAGE, 10.0),
]


class TestNumericalVerifier:
    @pytest.mark.parametrize("text,expected_type,expected_value", FIXTURES)
    def test_extract_claims(self, text, expected_type, expected_value):
        claims = extract_claims(text)
        if expected_type is None:
            assert claims == []
            return
        assert any(c.claim_type == expected_type and c.value == expected_value for c in claims)

    def test_verify_returns_status(self):
        claims = extract_claims("CVaR 12.3% drawdown 8.7%")
        statuses = [verify_claim(c) for c in claims]
        assert all(s in VerifyStatus for s in statuses)
