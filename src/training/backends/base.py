"""Training backend protocol — TRL vs Unsloth DPO paths."""

from __future__ import annotations

from typing import Any, Dict, List, Protocol, Tuple, runtime_checkable


@runtime_checkable
class TrainerBackend(Protocol):
    """Load a PEFT-wrapped model and run DPO training."""

    name: str

    def load_model(self, config: Dict[str, Any]) -> Tuple[Any, Any]:
        """Return ``(model, tokenizer)`` ready for DPO."""
        ...

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
        """Run DPO and return checkpoint directory path."""
        ...
