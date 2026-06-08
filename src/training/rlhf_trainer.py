"""
RLHF trainer using Direct Preference Optimization (DPO).

Wraps TRL's ``DPOTrainer`` with:
    * QLoRA model from :class:`~src.models.model_wrapper.ModelWrapper`
    * Preference pairs produced by :class:`~src.training.preference_builder.PreferenceBuilder`
    * Composite reward logging via :class:`~src.training.reward_composer.RewardComposer`
    * Optional Weights & Biases reporting

Memory budget for a 7B model on a single 24 GB GPU:
    * 4-bit NF4 base weights     ≈  4–5 GB
    * LoRA trainable parameters  ≈  0.5 GB
    * Optimizer states (8-bit)   ≈  1–2 GB
    * Activations + batch        ≈  8–12 GB  (batch_size=4, grad_accum=8)
    Remaining headroom           ≈  4–8 GB  ✓

DPO uses an implicit reference policy (disabled LoRA adapters), eliminating
the need for a separate frozen reference model copy.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


class RLHFTrainer:
    """
    End-to-end DPO trainer for the Scenario Reasoner LM.

    Args:
        model: PEFT causal LM (from :class:`~src.models.model_wrapper.ModelWrapper`).
        tokenizer: Corresponding tokenizer.
        preference_data: List of ``{"prompt", "chosen", "rejected"}`` dicts.
        config: Hyperparameter dict (typically loaded from
                ``experiments/configs/causal_rlhf_config.json``).
        use_wandb: Enable Weights & Biases logging.
        output_dir: Directory for checkpoints and TRL logs.

    Example::

        trainer = RLHFTrainer(model, tokenizer, preference_data, config)
        checkpoint_path = trainer.train()
    """

    def __init__(
        self,
        model,
        tokenizer,
        preference_data: List[Dict[str, str]],
        config: Dict[str, Any],
        use_wandb: bool = False,
        output_dir: str = "experiments/results",
    ) -> None:
        self.model = model
        self.tokenizer = tokenizer
        self.preference_data = preference_data
        self.config = config
        self.use_wandb = use_wandb
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def train(self) -> str:
        """
        Run DPO training on the preference dataset.

        Returns:
            Absolute path to the saved checkpoint directory.

        Raises:
            ImportError: If ``trl>=0.8`` is not installed.
            ValueError: If ``preference_data`` is empty.
        """
        if not self.preference_data:
            raise ValueError("preference_data is empty — cannot start training.")

        try:
            import datasets as hf_datasets
            from trl import DPOConfig, DPOTrainer
        except ImportError as exc:
            raise ImportError(
                "RLHFTrainer requires 'trl>=0.8'. "
                "Install with: pip install 'trl>=0.8'"
            ) from exc

        logger.info(
            "Building HuggingFace dataset from %d preference pairs",
            len(self.preference_data),
        )
        hf_dataset = hf_datasets.Dataset.from_list(self.preference_data)

        dpo_config = self._build_dpo_config()
        trainer_callbacks = self._build_callbacks()

        trainer = DPOTrainer(
            model=self.model,
            ref_model=None,
            args=dpo_config,
            train_dataset=hf_dataset,
            tokenizer=self.tokenizer,
            beta=self.config.get("dpo_beta", 0.1),
            max_length=self.config.get("max_seq_length", 2048),
            max_prompt_length=self.config.get("max_prompt_length", 1024),
            callbacks=trainer_callbacks,
        )

        logger.info("Starting DPO training — %d epochs", self.config.get("num_epochs", 3))
        trainer.train()

        self._log_callback_outputs()

        checkpoint_path = str(self.output_dir / "dpo_checkpoint")
        trainer.save_model(checkpoint_path)
        self.tokenizer.save_pretrained(checkpoint_path)
        logger.info("Checkpoint saved to: %s", checkpoint_path)
        return checkpoint_path

    # ── Internals ─────────────────────────────────────────────────────────────

    def _build_callbacks(self) -> list:
        training_cfg = self.config.get("training", {})
        monitor = training_cfg.get("monitor_gates") or self.config.get("monitor_gates")
        if not monitor:
            return []

        from transformers import TrainerCallback
        from src.training.callbacks.reward_theta_callbacks import RewardDecompositionCallback

        reward_cb = RewardDecompositionCallback()
        self._reward_callback = reward_cb
        local_logger = self.config.get("_local_logger")

        class _RewardLogCallback(TrainerCallback):
            def on_log(self, args, state, control, logs=None, **kwargs):
                if logs:
                    breakdown = {
                        k: float(v)
                        for k, v in logs.items()
                        if isinstance(v, (int, float))
                    }
                    if breakdown:
                        reward_cb.on_step_end(breakdown)

        callbacks: list = [_RewardLogCallback()]
        if training_cfg.get("early_stop_theta_stratified"):
            from src.training.callbacks.reward_theta_callbacks import ThetaStratifiedEarlyStopCallback
            self._theta_stop_callback = ThetaStratifiedEarlyStopCallback()
            callbacks.append(self._theta_stop_callback)  # type: ignore[arg-type]
        if local_logger is not None:
            self._callback_local_logger = local_logger
        return callbacks

    def _log_callback_outputs(self) -> None:
        reward_cb = getattr(self, "_reward_callback", None)
        local_logger = getattr(self, "_callback_local_logger", None)
        if reward_cb is None or local_logger is None:
            return
        agg = reward_cb.aggregate()
        if agg:
            local_logger.log_step(step=0, metrics=agg, prefix="callback/reward_decomposition")

    def _build_dpo_config(self):
        """Construct a TRL ``DPOConfig`` from the project config dict."""
        from trl import DPOConfig

        return DPOConfig(
            output_dir=str(self.output_dir),
            per_device_train_batch_size=self.config.get("batch_size", 4),
            gradient_accumulation_steps=self.config.get("gradient_accumulation", 8),
            learning_rate=self.config.get("learning_rate", 1e-4),
            num_train_epochs=self.config.get("num_epochs", 3),
            lr_scheduler_type=self.config.get("lr_scheduler", "cosine"),
            warmup_ratio=self.config.get("warmup_ratio", 0.1),
            fp16=self.config.get("fp16", False),
            bf16=self.config.get("bf16", True),
            logging_steps=self.config.get("logging_steps", 10),
            save_steps=self.config.get("save_steps", 100),
            report_to="wandb" if self.use_wandb else "none",
            remove_unused_columns=False,
            optim=self.config.get("optimizer", "paged_adamw_8bit"),
        )
