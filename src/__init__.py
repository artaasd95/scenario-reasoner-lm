"""
Scenario Reasoner LM

A training suite for open-source reasoning LLMs to reason into scenarios.
Provides base infrastructure for metrics, losses, datasets, monitoring, and logging.

Sub-packages are imported lazily so that pure-Python modules (scenarios,
metrics, reward functions) remain importable even when optional heavy
dependencies (torch, bitsandbytes, peft) are not installed.
"""

__version__ = "0.1.0"
__author__ = "artaasd95"


def __getattr__(name):
    """Lazy sub-package access — ``from src import data`` still works."""
    import importlib
    try:
        module = importlib.import_module(f"src.{name}")
        globals()[name] = module
        return module
    except ImportError as exc:
        raise AttributeError(
            f"module 'src' has no attribute {name!r} "
            f"(underlying import failed: {exc})"
        ) from exc
