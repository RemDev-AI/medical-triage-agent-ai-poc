# medical-triage-agent-ai-poc/backend/app/training/shared/training_model_loader.py
# model = setup_peft_model(
from __future__ import annotations

import logging
from typing import Any
from typing import Dict
from typing import Optional  # noqa : F401

import torch
from transformers import AutoModelForCausalLM

from backend.app.training.lora.peft_setup import setup_peft_model

logger = logging.getLogger(__name__)


class TrainingModelLoader:
    """
    Shared model loader used by all training pipelines.

    Supported pipelines:
        - SFT
        - DPO

    Responsibilities:
        - Load base model
        - Configure torch dtype
        - Enable gradient checkpointing
        - Apply PEFT / LoRA
        - Prepare model for training

    Non-responsibilities:
        - Tokenizer loading
        - Dataset loading
        - WandB
        - Checkpoints
        - HF Hub uploads
        - Clinical evaluation
    """

    def __init__(self, config: Dict[str, Any]) -> None:
        self.config = config

    def load_base_model(self) -> AutoModelForCausalLM:
        """
        Load the base causal language model.
        """

        model_name = self.config["model"]["base_model"]

        logger.info(
            "Loading base model for training: %s",
            model_name,
        )

        model = AutoModelForCausalLM.from_pretrained(
            pretrained_model_name_or_path=model_name,
            torch_dtype=self._resolve_torch_dtype(),
            trust_remote_code=self.config["model"].get(
                "trust_remote_code",
                True,
            ),
            device_map=self.config["model"].get(
                "device_map",
                "auto",
            ),
        )

        logger.info("Base model loaded successfully")

        return model

    def apply_gradient_checkpointing(
        self,
        model: AutoModelForCausalLM,
    ) -> AutoModelForCausalLM:
        """
        Enable gradient checkpointing if configured.
        """

        enabled = self.config["training"].get(
            "gradient_checkpointing",
            True,
        )

        if enabled:
            logger.info("Enabling gradient checkpointing")

            model.gradient_checkpointing_enable()

            if hasattr(model, "enable_input_require_grads"):
                model.enable_input_require_grads()

        return model

    def apply_lora(
        self,
        model: AutoModelForCausalLM,
    ) -> AutoModelForCausalLM:
        """
        Apply LoRA adapters through PEFT setup.
        """

        logger.info("Applying LoRA adapters")

        model = setup_peft_model(
            model=model,
            # config=self.config,
        )

        logger.info("LoRA adapters applied successfully")

        return model

    def prepare_for_training(self) -> AutoModelForCausalLM:
        """
        Complete training preparation pipeline.

        Returns:
            PEFT model ready for SFT or DPO training.
        """

        model = self.load_base_model()

        model = self.apply_gradient_checkpointing(
            model=model,
        )

        model = self.apply_lora(
            model=model,
        )

        self._log_trainable_parameters(model)

        return model

    def _resolve_torch_dtype(self) -> torch.dtype:
        """
        Resolve dtype from configuration.
        """

        dtype = self.config["model"].get(
            "torch_dtype",
            "bfloat16",
        )

        mapping = {
            "float16": torch.float16,
            "fp16": torch.float16,
            "bfloat16": torch.bfloat16,
            "bf16": torch.bfloat16,
            "float32": torch.float32,
            "fp32": torch.float32,
        }

        resolved_dtype = mapping.get(
            dtype.lower(),
            torch.bfloat16,
        )

        logger.info(
            "Using torch dtype: %s",
            resolved_dtype,
        )

        return resolved_dtype

    @staticmethod
    def _log_trainable_parameters(
        model: AutoModelForCausalLM,
    ) -> None:
        """
        Log trainable parameters after LoRA injection.
        """

        trainable_params = 0
        total_params = 0

        for parameter in model.parameters():
            total_params += parameter.numel()

            if parameter.requires_grad:
                trainable_params += parameter.numel()

        percentage = (
            100 * trainable_params / total_params
            if total_params > 0
            else 0.0
        )

        logger.info(
            "Trainable parameters: %s / %s (%.4f%%)",
            f"{trainable_params:,}",
            f"{total_params:,}",
            percentage,
        )

    @classmethod
    def build(
        cls,
        config: Dict[str, Any],
    ) -> AutoModelForCausalLM:
        """
        Convenience entrypoint.

        Example:
            model = TrainingModelLoader.build(config)
        """

        return cls(config).prepare_for_training()
