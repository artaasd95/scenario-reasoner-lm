"""Unsloth FastLanguageModel + TRL DPO backend."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict, List, Tuple

logger = logging.getLogger(__name__)


def _build_dpo_config(config: Dict[str, Any], output_dir: str, use_wandb: bool):
    from trl import DPOConfig

    return DPOConfig(
        output_dir=output_dir,
        per_device_train_batch_size=config.get("batch_size", 4),
        gradient_accumulation_steps=config.get("gradient_accumulation", 8),
        learning_rate=config.get("learning_rate", 1e-4),
        num_train_epochs=config.get("num_epochs", 3),
        lr_scheduler_type=config.get("lr_scheduler", "cosine"),
        warmup_ratio=config.get("warmup_ratio", 0.1),
        fp16=config.get("fp16", False),
        bf16=config.get("bf16", True),
        logging_steps=config.get("logging_steps", 10),
        save_strategy="steps",
        save_steps=config.get("save_steps", 100),
        save_total_limit=config.get("save_total_limit", 3),
        resume_from_checkpoint=config.get("resume_from_checkpoint") or None,
        report_to="wandb" if use_wandb else "none",
        remove_unused_columns=False,
        optim=config.get("optimizer", "paged_adamw_8bit"),
    )


class UnslothBackend:
    """Load via Unsloth FastLanguageModel and train with TRL DPOTrainer."""

    name = "unsloth"

    def load_model(self, config: Dict[str, Any]) -> Tuple[Any, Any]:
        try:
            from unsloth import FastLanguageModel
        except ImportError as exc:
            raise ImportError(
                "Unsloth backend requires the optional extra. "
                "Install with: pip install -e '.[unsloth]'"
            ) from exc

        model_path = config["model_name_or_path"]
        max_seq_length = config.get("max_seq_length", 2048)
        use_qlora = config.get("use_qlora", True)

        logger.info("Loading model via Unsloth: %s", model_path)
        model, tokenizer = FastLanguageModel.from_pretrained(
            model_name=model_path,
            max_seq_length=max_seq_length,
            dtype=None,
            load_in_4bit=use_qlora,
            trust_remote_code=True,
        )

        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        tokenizer.padding_side = "right"

        target_modules = config.get(
            "target_modules",
            ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        )
        model = FastLanguageModel.get_peft_model(
            model,
            r=config.get("lora_r", 16),
            target_modules=target_modules,
            lora_alpha=config.get("lora_alpha", 32),
            lora_dropout=config.get("lora_dropout", 0.05),
            bias="none",
            use_gradient_checkpointing="unsloth",
            random_state=config.get("data", {}).get("seed", 42),
        )
        return model, tokenizer

    def train(
        self,
        model: Any,
        tokenizer: Any,
        preference_data: List[Dict[str, str]],
        config: Dict[str, Any],
        *,
        use_wandb: bool = False,
        output_dir: str = "experiments/results",
    ) -> str:
        if not preference_data:
            raise ValueError("preference_data is empty — cannot start training.")

        try:
            import datasets as hf_datasets
            from trl import DPOTrainer
        except ImportError as exc:
            raise ImportError(
                "Unsloth backend requires 'trl>=0.8'. "
                "Install with: pip install 'trl>=0.8'"
            ) from exc

        out = Path(output_dir)
        out.mkdir(parents=True, exist_ok=True)

        hf_dataset = hf_datasets.Dataset.from_list(preference_data)
        dpo_config = _build_dpo_config(config, str(out), use_wandb)

        callbacks = []
        training_cfg = config.get("training", {})
        if training_cfg.get("monitor_gates"):
            from src.training.callbacks.reward_theta_callbacks import RewardDecompositionCallback

            reward_cb = RewardDecompositionCallback()
            callbacks.append(reward_cb)

        trainer = DPOTrainer(
            model=model,
            ref_model=None,
            args=dpo_config,
            train_dataset=hf_dataset,
            processing_class=tokenizer,
            beta=config.get("dpo_beta", 0.1),
            max_length=config.get("max_seq_length", 2048),
            max_prompt_length=config.get("max_prompt_length", 1024),
            callbacks=callbacks,
        )

        logger.info(
            "Starting Unsloth DPO training — %d pairs, %d epochs",
            len(preference_data),
            config.get("num_epochs", 3),
        )
        trainer.train()

        local_logger = config.get("_local_logger")
        if local_logger and callbacks:
            from src.training.callbacks.reward_theta_callbacks import RewardDecompositionCallback
            for cb in callbacks:
                if isinstance(cb, RewardDecompositionCallback):
                    agg = cb.aggregate()
                    if agg:
                        local_logger.log_step(
                            step=0, metrics=agg, prefix="callback/reward_decomposition"
                        )

        checkpoint_path = str(out / "dpo_checkpoint")
        trainer.save_model(checkpoint_path)
        tokenizer.save_pretrained(checkpoint_path)
        logger.info("Unsloth checkpoint saved to: %s", checkpoint_path)
        return checkpoint_path
