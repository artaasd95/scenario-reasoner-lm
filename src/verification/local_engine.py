"""NumPy-only local verification engine (SR-20)."""

from __future__ import annotations

from enum import Enum
from typing import Optional

import numpy as np

from src.verification.numerical_verifier import ClaimType, ExtractedClaim


class VerifyStatus(str, Enum):
    VERIFIED = "VERIFIED"
    FLAGGED = "FLAGGED"
    UNVERIFIED = "UNVERIFIED"


def cvar_normal_approx(returns: np.ndarray, alpha: float = 0.05) -> float:
    mu = float(np.mean(returns))
    sigma = float(np.std(returns))
    if sigma < 1e-9:
        return abs(mu) * 100
    # approximate z for 5% tail
    z = -1.645
    pdf_z = np.exp(-0.5 * z * z) / np.sqrt(2 * np.pi)
    cvar = mu + sigma * pdf_z / alpha
    return abs(cvar) * 100


def max_drawdown_pct(prices: np.ndarray) -> float:
    peak = np.maximum.accumulate(prices)
    dd = (peak - prices) / np.maximum(peak, 1e-9)
    return float(np.max(dd)) * 100


def leverage_ratio(assets: float, equity: float) -> float:
    if equity <= 0:
        return float("inf")
    return assets / equity


def verify_claim(
    claim: ExtractedClaim,
    *,
    returns: Optional[np.ndarray] = None,
    prices: Optional[np.ndarray] = None,
    assets: float = 100.0,
    equity: float = 50.0,
    verified_tol: float = 0.05,
    flag_tol: float = 0.20,
) -> VerifyStatus:
    if claim.value is None:
        return VerifyStatus.UNVERIFIED

    if claim.claim_type == ClaimType.CVAR:
        if returns is None:
            returns = np.random.default_rng(42).normal(0, 0.02, 100)
        expected = cvar_normal_approx(returns)
        rel_err = abs(claim.value - expected) / max(expected, 1e-6)
        if rel_err <= verified_tol:
            return VerifyStatus.VERIFIED
        if rel_err <= flag_tol:
            return VerifyStatus.FLAGGED
        return VerifyStatus.FLAGGED

    if claim.claim_type == ClaimType.DRAWDOWN:
        if prices is None:
            prices = np.cumprod(1 + np.random.default_rng(7).normal(0, 0.01, 50))
        expected = max_drawdown_pct(prices)
        rel_err = abs(claim.value - expected) / max(expected, 1e-6)
        if rel_err <= verified_tol:
            return VerifyStatus.VERIFIED
        if rel_err <= flag_tol:
            return VerifyStatus.FLAGGED
        return VerifyStatus.FLAGGED

    if claim.claim_type == ClaimType.LEVERAGE:
        expected = leverage_ratio(assets, equity)
        rel_err = abs(claim.value - expected) / max(expected, 1e-6)
        if rel_err <= verified_tol:
            return VerifyStatus.VERIFIED
        if rel_err <= flag_tol:
            return VerifyStatus.FLAGGED
        return VerifyStatus.FLAGGED

    return VerifyStatus.UNVERIFIED
