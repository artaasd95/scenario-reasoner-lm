"""
QLoRA model wrapper for 7B parameter causal language models.

Handles:
    * 4-bit NF4 quantization via ``bitsandbytes``
    * LoRA adapter injection via ``peft``
    * Tokenizer configuration (pad token, padding side)
    * Device placement via ``device_map``

This module has **no hard imports** of heavy libraries at the module level —
all imports are deferred to :meth:`ModelWrapper.load` so the rest of the project
remains importable even when CUDA packages are absent (e.g. in CI or evaluation-
only environments).

Example::

    from src.models.model_wrapper import ModelWrapper

    wrapper = ModelWrapper(
        model_name_or_path="mistralai/Mistral-7B-Instruct-v0.2",
        lora_r=16,
        lora_alpha=32,
        use_qlora=True,
    )
    model, tokenizer = wrapper.load()
"""

from __future__ import annotations

import logging
from typing import List, Optional, Tuple

logger = logging.getLogger(__name__)


class ModelWrapper:
    """
    Loads a causal language model with optional QLoRA for fine-tuning.

    The loaded model is returned as a PEFT-wrapped ``PeftModelForCausalLM``
    with LoRA adapters targeting the attention projection layers.  The base
    model weights are kept in 4-bit NF4 quantization when ``use_qlora=True``.

    Args:
        model_name_or_path: HuggingFace model hub identifier or local path.
            Recommended: ``"mistralai/Mistral-7B-Instruct-v0.2"`` or
            ``"meta-llama/Llama-3.1-7B-Instruct"``.
        lora_r: LoRA rank (number of learnable parameters per layer pair).
            Default ``16`` balances quality and memory.
        lora_alpha: LoRA scaling factor.  Typically ``2 * lora_r``.
        lora_dropout: Dropout probability on LoRA layers.
        target_modules: List of linear module names to apply LoRA to.
            Defaults to ``["q_proj", "v_proj", "k_proj", "o_proj"]``.
        use_qlora: Load model in 4-bit NF4 quantization.  Required for
            fine-tuning a 7B model on ≤24 GB VRAM.
        device_map: HuggingFace device placement strategy.
            ``"auto"`` distributes layers across available GPUs.
        max_seq_length: Maximum token sequence length used to set
            ``tokenizer.model_max_length``.

    Example::

        wrapper = ModelWrapper("mistralai/Mistral-7B-Instruct-v0.2")
        model, tokenizer = wrapper.load()

        # Inspect trainable parameter count
        model.print_trainable_parameters()
    """

    DEFAULT_TARGET_MODULES: List[str] = ["q_proj", "v_proj", "k_proj", "o_proj"]

    def __init__(
        self,
        model_name_or_path: str,
        lora_r: int = 16,
        lora_alpha: int = 32,
        lora_dropout: float = 0.05,
        target_modules: Optional[List[str]] = None,
        use_qlora: bool = True,
        device_map: str = "auto",
        max_seq_length: int = 2048,
    ) -> None:
        self.model_name_or_path = model_name_or_path
        self.lora_r = lora_r
        self.lora_alpha = lora_alpha
        self.lora_dropout = lora_dropout
        self.target_modules = target_modules or self.DEFAULT_TARGET_MODULES
        self.use_qlora = use_qlora
        self.device_map = device_map
        self.max_seq_length = max_seq_length

    def load(self) -> Tuple:
        """
        Load and return ``(model, tokenizer)``.

        The model is 4-bit NF4 quantized (when ``use_qlora=True``) and wrapped
        with LoRA adapters.  Only the LoRA parameters are set to require
        gradients; the quantized base weights are frozen.

        Returns:
            Tuple ``(PeftModelForCausalLM, PreTrainedTokenizer)``.

        Raises:
            ImportError: If ``transformers``, ``peft``, or ``bitsandbytes``
                         are not installed.
        """
        try:
            import torch
            from peft import LoraConfig, TaskType, get_peft_model, prepare_model_for_kbit_training
            from transformers import (
                AutoModelForCausalLM,
                AutoTokenizer,
                BitsAndBytesConfig,
            )
        except ImportError as exc:
            raise ImportError(
                "ModelWrapper requires 'transformers', 'peft', 'bitsandbytes', and 'accelerate'. "
                "Install with: pip install transformers peft bitsandbytes accelerate"
            ) from exc

        logger.info("Loading tokenizer: %s", self.model_name_or_path)
        tokenizer = AutoTokenizer.from_pretrained(
            self.model_name_or_path,
            trust_remote_code=True,
        )
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        tokenizer.padding_side = "right"
        tokenizer.model_max_length = self.max_seq_length

        model_kwargs = {
            "trust_remote_code": True,
            "device_map": self.device_map,
        }

        if self.use_qlora:
            logger.info("Applying 4-bit NF4 QLoRA quantization")
            bnb_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=torch.bfloat16,
                bnb_4bit_use_double_quant=True,
            )
            model_kwargs["quantization_config"] = bnb_config

        logger.info("Loading model: %s", self.model_name_or_path)
        model = AutoModelForCausalLM.from_pretrained(
            self.model_name_or_path,
            **model_kwargs,
        )
        model.config.use_cache = False

        if self.use_qlora:
            model = prepare_model_for_kbit_training(model)

        lora_config = LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            r=self.lora_r,
            lora_alpha=self.lora_alpha,
            lora_dropout=self.lora_dropout,
            target_modules=self.target_modules,
            bias="none",
        )
        model = get_peft_model(model, lora_config)
        model.print_trainable_parameters()

        return model, tokenizer

    @classmethod
    def from_config(cls, config: dict) -> "ModelWrapper":
        """
        Construct a :class:`ModelWrapper` from a config dictionary.

        Recognises the following keys (all optional except ``model_name_or_path``):
        ``lora_r``, ``lora_alpha``, ``lora_dropout``, ``target_modules``,
        ``use_qlora``, ``device_map``, ``max_seq_length``.

        Args:
            config: Dict typically loaded from a JSON config file.

        Returns:
            A configured :class:`ModelWrapper` instance.
        """
        return cls(
            model_name_or_path=config["model_name_or_path"],
            lora_r=config.get("lora_r", 16),
            lora_alpha=config.get("lora_alpha", 32),
            lora_dropout=config.get("lora_dropout", 0.05),
            target_modules=config.get("target_modules"),
            use_qlora=config.get("use_qlora", True),
            device_map=config.get("device_map", "auto"),
            max_seq_length=config.get("max_seq_length", 2048),
        )
