"""Shared TRL DPOConfig builder for training backends."""

from __future__ import annotations

from typing import Any, Dict, Optional


def build_dpo_config(
    config: Dict[str, Any],
    output_dir: str,
    *,
    use_wandb: bool = False,
    save_strategy: Optional[str] = None,
    save_total_limit: Optional[int] = None,
    resume_from_checkpoint: Optional[str] = None,
):
    """Construct a TRL ``DPOConfig`` shared by TRL and Unsloth backends."""
    from trl import DPOConfig

    kwargs: Dict[str, Any] = {
        "output_dir": output_dir,
        "per_device_train_batch_size": config.get("batch_size", 4),
        "gradient_accumulation_steps": config.get("gradient_accumulation", 8),
        "learning_rate": config.get("learning_rate", 1e-4),
        "num_train_epochs": config.get("num_epochs", 3),
        "lr_scheduler_type": config.get("lr_scheduler", "cosine"),
        "warmup_ratio": config.get("warmup_ratio", 0.1),
        "fp16": config.get("fp16", False),
        "bf16": config.get("bf16", True),
        "logging_steps": config.get("logging_steps", 10),
        "save_steps": config.get("save_steps", 100),
        "report_to": "wandb" if use_wandb else "none",
        "remove_unused_columns": False,
        "optim": config.get("optimizer", "paged_adamw_8bit"),
    }
    if save_strategy is not None:
        kwargs["save_strategy"] = save_strategy
    if save_total_limit is not None:
        kwargs["save_total_limit"] = save_total_limit
    if resume_from_checkpoint is not None:
        kwargs["resume_from_checkpoint"] = resume_from_checkpoint or None
    return DPOConfig(**kwargs)
