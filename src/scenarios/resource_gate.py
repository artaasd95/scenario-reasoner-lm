"""
Resource gates for live provider and paid API usage (S5-02, S5-06, S7-06).

Default development and CI remain mock/smoke/offline unless an explicit gate
is set.
"""

from __future__ import annotations

import os


ALLOW_LIVE_PROVIDER_ENV = "ALLOW_LIVE_PROVIDER"
LLM_PROVIDER_ENV = "LLM_PROVIDER"
EXECUTION_SPRINT_GATE_ENV = "EXECUTION_SPRINT_GATE"


def allow_live_provider() -> bool:
    """True when ``ALLOW_LIVE_PROVIDER=1`` (or ``true``/``yes``)."""
    raw = os.getenv(ALLOW_LIVE_PROVIDER_ENV, "").strip().lower()
    return raw in ("1", "true", "yes")


def execution_sprint_allowed() -> bool:
    """True when ``EXECUTION_SPRINT_GATE=1`` (or ``true``/``yes``)."""
    raw = os.getenv(EXECUTION_SPRINT_GATE_ENV, "").strip().lower()
    return raw in ("1", "true", "yes")


def llm_provider_offline() -> bool:
    return os.getenv(LLM_PROVIDER_ENV, "offline").strip().lower() == "offline"


def assert_mock_or_gated(*, live_requested: bool = False) -> None:
    """
    Block live/paid provider calls unless the resource gate is set.

    Raises:
        RuntimeError: When live is requested but ``ALLOW_LIVE_PROVIDER`` is unset.
    """
    if live_requested and not allow_live_provider():
        raise RuntimeError(
            f"Live provider calls are blocked. Set {ALLOW_LIVE_PROVIDER_ENV}=1 "
            f"and ensure budget approval before running paid API or GPU jobs."
        )


def assert_execution_sprint_allowed(*, full_pipeline: bool = False) -> None:
    """
    Block full exploratory/enumerated pipeline runs unless execution sprint gate is set.

    Raises:
        RuntimeError: When full pipeline is requested but gate is unset.
    """
    if full_pipeline and not execution_sprint_allowed():
        raise RuntimeError(
            f"Full pipeline execution is blocked. Set {EXECUTION_SPRINT_GATE_ENV}=1 "
            f"during an approved execution sprint window. Dev/CI default is unit/smoke on fixtures."
        )


def effective_provider_mode(*, live_requested: bool = False) -> str:
    """Return ``mock``, ``offline``, or ``live`` for run metadata."""
    if live_requested and allow_live_provider() and not llm_provider_offline():
        return "live"
    if live_requested and allow_live_provider():
        return "offline"
    return "mock"
