"""Vanilla TRL + ModelWrapper QLoRA backend (default)."""

from __future__ import annotations

from typing import Any, Dict, List, Tuple

from src.models.model_wrapper import ModelWrapper
from src.training.rlhf_trainer import RLHFTrainer


class TrlBackend:
    name = "trl"

    def load_model(self, config: Dict[str, Any]) -> Tuple[Any, Any]:
        wrapper = ModelWrapper.from_config(config)
        return wrapper.load()

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
        trainer = RLHFTrainer(
            model=model,
            tokenizer=tokenizer,
            preference_data=preference_data,
            config=config,
            use_wandb=use_wandb,
            output_dir=output_dir,
        )
        return trainer.train()
