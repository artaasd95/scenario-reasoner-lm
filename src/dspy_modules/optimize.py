"""
DSPy optimizer selection for enterprise risk demo (S4-03).

BootstrapFewShot is the default. MIPRO is gated behind ``ENABLE_MIPRO=1`` or
explicit ``optimizer=MIPRO``; failures fall back to BootstrapFewShot and are traced.
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass
from typing import Any, Callable, Optional

logger = logging.getLogger(__name__)

OPTIMIZER_BOOTSTRAP = "BootstrapFewShot"
OPTIMIZER_MIPRO = "MIPRO"

# Documented equal budget controls for baseline vs MIPRO comparison
DEFAULT_BOOTSTRAP_BUDGET = {
    "max_bootstrapped_demos": 4,
    "max_labeled_demos": 8,
}
DEFAULT_MIPRO_BUDGET = {
    "max_bootstrapped_demos": 4,
    "max_labeled_demos": 8,
    "num_candidates": 10,
    "init_temperature": 1.0,
}


@dataclass
class OptimizerConfig:
    name: str = OPTIMIZER_BOOTSTRAP
    seed: int = 42
    bootstrap_budget: dict = None  # type: ignore
    mipro_budget: dict = None  # type: ignore
    enable_mipro: bool = False

    def __post_init__(self) -> None:
        if self.bootstrap_budget is None:
            self.bootstrap_budget = dict(DEFAULT_BOOTSTRAP_BUDGET)
        if self.mipro_budget is None:
            self.mipro_budget = dict(DEFAULT_MIPRO_BUDGET)

    @property
    def token_budget_note(self) -> str:
        if self.name == OPTIMIZER_MIPRO:
            return f"MIPRO budget: {self.mipro_budget}"
        return f"BootstrapFewShot budget: {self.bootstrap_budget}"


def mipro_enabled(explicit_optimizer: Optional[str] = None) -> bool:
    if explicit_optimizer == OPTIMIZER_MIPRO:
        return True
    return os.getenv("ENABLE_MIPRO", "").strip() in ("1", "true", "yes")


def resolve_optimizer(
    explicit: Optional[str] = None,
    *,
    seed: int = 42,
) -> OptimizerConfig:
    """Resolve optimizer from CLI/config with MIPRO feature flag."""
    requested = (explicit or OPTIMIZER_BOOTSTRAP).strip()
    if requested == OPTIMIZER_MIPRO and mipro_enabled(explicit_optimizer=OPTIMIZER_MIPRO):
        return OptimizerConfig(name=OPTIMIZER_MIPRO, seed=seed, enable_mipro=True)
    if requested == OPTIMIZER_MIPRO and not mipro_enabled(explicit_optimizer=OPTIMIZER_MIPRO):
        logger.warning("MIPRO requested but ENABLE_MIPRO is not set; using BootstrapFewShot")
    return OptimizerConfig(name=OPTIMIZER_BOOTSTRAP, seed=seed, enable_mipro=False)


def _require_dspy():
    try:
        import dspy  # noqa: F401
    except ImportError as exc:
        raise ImportError("dspy is not installed. pip install dspy-ai") from exc


def build_optimizer(config: OptimizerConfig, metric: Optional[Callable] = None) -> Any:
    """
    Return a DSPy teleprompter instance.

    In offline/dev mode callers may skip compilation and only record ``config.name``.
    """
    _require_dspy()
    import dspy
    from dspy.teleprompt import BootstrapFewShot

    metric_fn = metric or (lambda *a, **k: 1.0)

    if config.name == OPTIMIZER_MIPRO and config.enable_mipro:
        try:
            from dspy.teleprompt import MIPRO

            return MIPRO(
                metric=metric_fn,
                seed=config.seed,
                **config.mipro_budget,
            )
        except Exception as exc:  # pragma: no cover - import/runtime failures
            logger.exception("MIPRO init failed; falling back to BootstrapFewShot: %s", exc)
            config.name = OPTIMIZER_BOOTSTRAP
            config.enable_mipro = False

    return BootstrapFewShot(
        metric=metric_fn,
        **config.bootstrap_budget,
    )


def compile_module(module: Any, optimizer_config: OptimizerConfig, trainset: list) -> Any:
    """
    Compile ``module`` with the resolved optimizer; non-fatal on failure.
    """
    if not trainset:
        logger.info("Empty trainset; skipping optimizer compile")
        return module
    try:
        teleprompter = build_optimizer(optimizer_config)
        return teleprompter.compile(module, trainset=trainset)
    except Exception as exc:
        logger.exception("Optimizer compile failed (non-fatal): %s", exc)
        return module
