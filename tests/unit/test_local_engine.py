"""Local verification engine tests (SR-20)."""

from __future__ import annotations

import numpy as np

from src.verification.local_engine import (
    cvar_normal_approx,
    leverage_ratio,
    max_drawdown_pct,
)
from src.verification.numerical_verifier import ClaimType, ExtractedClaim
from src.verification.local_engine import verify_claim, VerifyStatus


class TestLocalEngine:
    def test_cvar_known_answer(self):
        rng = np.random.default_rng(0)
        returns = rng.normal(0, 0.02, 200)
        val = cvar_normal_approx(returns)
        assert 0 < val < 100

    def test_max_drawdown(self):
        prices = np.array([1.0, 1.1, 0.9, 0.85, 0.95])
        dd = max_drawdown_pct(prices)
        assert abs(dd - 22.727) < 5.0

    def test_leverage_ratio(self):
        assert leverage_ratio(100, 50) == 2.0

    def test_verify_claim_tolerance(self):
        claim = ExtractedClaim(text="leverage 2.0", claim_type=ClaimType.LEVERAGE, value=2.0)
        status = verify_claim(claim, assets=100, equity=50)
        assert status == VerifyStatus.VERIFIED
