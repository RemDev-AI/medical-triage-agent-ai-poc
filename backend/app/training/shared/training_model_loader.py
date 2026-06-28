# medical-triage-agent-ai-poc/backend/app/training/shared/training_model_loader.py
# Corrections : bug #2 (config LoRA passée), bug #3 (torch_dtype auto),
#               bug #4 (use_reentrant=False pour Qwen3)

from __future__ import annotations

import logging
from typing import Any, Dict, Optional  # noqa: F401

import torch
from transformers import AutoModelForCausalLM

from backend.app.training.lora.peft_setup import setup_peft_model

logger = logging.getLogger(__name__)


class TrainingModelLoader:
    """
    Shared model loader — SFT & DPO pipelines.
    """

    def __init__(self, config: Dict[str, Any]) -> None:
        self.config = config

    def load_base_model(self) -> AutoModelForCausalLM:
        model_name = self.config["model"]["base_model"]

        logger.info("Loading base model: %s", model_name)

        model = AutoModelForCausalLM.from_pretrained(
            pretrained_model_name_or_path=model_name,
            torch_dtype=self._resolve_torch_dtype(),
            trust_remote_code=self.config["model"].get("trust_remote_code", True),  # noqa: E501
            device_map=self.config["model"].get("device_map", "auto"),
        )

        logger.info("Base model loaded successfully.")
        return model

    def apply_gradient_checkpointing(
        self,
        model: AutoModelForCausalLM,
    ) -> AutoModelForCausalLM:
        enabled = self.config["training"].get("gradient_checkpointing", True)

        if enabled:
            logger.info("Enabling gradient checkpointing (use_reentrant=False).")  # noqa: E501

            # FIX BUG #4 — use_reentrant=False requis pour Qwen3
            # + DataCollatorForSeq2Seq (évite NaN silencieux sur certaines versions)  # noqa: E501
            model.gradient_checkpointing_enable(
                gradient_checkpointing_kwargs={"use_reentrant": False}
            )

            if hasattr(model, "enable_input_require_grads"):
                model.enable_input_require_grads()

        return model

    def apply_lora(
        self,
        model: AutoModelForCausalLM,
    ) -> AutoModelForCausalLM:
        logger.info("Applying LoRA adapters.")

        # FIX BUG #2 — config passée à setup_peft_model (était commentée)
        model = setup_peft_model(
            model=model,
            config=self.config,     # ← décommenté : la config YAML est lue
        )

        logger.info("LoRA adapters applied successfully.")
        return model

    def prepare_for_training(self) -> AutoModelForCausalLM:
        model = self.load_base_model()
        model = self.apply_gradient_checkpointing(model=model)
        model = self.apply_lora(model=model)
        self._log_trainable_parameters(model)
        return model

    # FIX BUG #3 — torch_dtype "auto" délégué à colab_environment
    def _resolve_torch_dtype(self) -> torch.dtype:
        dtype = self.config["model"].get("torch_dtype", "auto")

        if dtype == "auto":
            # Source unique de vérité : même logique que apply_precision_arguments()  # noqa: E501
            from backend.app.training.colab.colab_environment import (
                get_training_dtype,
            )
            resolved = get_training_dtype()
            logger.info("torch_dtype=auto → résolu par runtime : %s", resolved)
            return resolved

        mapping = {
            "float16": torch.float16,
            "fp16": torch.float16,
            "bfloat16": torch.bfloat16,
            "bf16": torch.bfloat16,
            "float32": torch.float32,
            "fp32": torch.float32,
        }
        resolved = mapping.get(dtype.lower(), torch.float16)
        logger.info("torch_dtype=%s → %s", dtype, resolved)
        return resolved

    @staticmethod
    def _log_trainable_parameters(model: AutoModelForCausalLM) -> None:
        trainable_params = 0
        total_params = 0

        for parameter in model.parameters():
            total_params += parameter.numel()
            if parameter.requires_grad:
                trainable_params += parameter.numel()

        percentage = (
            100 * trainable_params / total_params if total_params > 0 else 0.0
        )

        logger.info(
            "Trainable parameters: %s / %s (%.4f%%)",
            f"{trainable_params:,}",
            f"{total_params:,}",
            percentage,
        )

    @classmethod
    def build(cls, config: Dict[str, Any]) -> AutoModelForCausalLM:
        return cls(config).prepare_for_training()
