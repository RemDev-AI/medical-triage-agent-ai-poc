# medical-triage-agent-ai-poc/backend/app/training/lora/lora_config.py

"""
Centralized LoRA configuration.

This module defines:
- LoRA hyperparameters
- target modules
- quantization compatibility
- reproducible configuration objects
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Dict, Optional

from peft import LoraConfig, TaskType

QWEN_LORA_TARGET_MODULES = (
    "q_proj",
    "k_proj",
    "v_proj",
    "o_proj",
    "gate_proj",
    "up_proj",
    "down_proj",
)


@dataclass
class LoRAHyperParameters:
    """
    Dataclass containing LoRA hyperparameters.
    """

    rank: int = 16
    alpha: int = 32
    dropout: float = 0.05

    bias: str = "none"

    task_type: TaskType = TaskType.CAUSAL_LM

    target_modules: tuple[str, ...] = field(
        default_factory=lambda: QWEN_LORA_TARGET_MODULES
    )

    modules_to_save: Optional[tuple[str, ...]] = field(
        default_factory=lambda: ("lm_head",)
    )

    inference_mode: bool = False

    use_rslora: bool = False
    use_dora: bool = False

    init_lora_weights: str = "gaussian"

    def validate(self) -> None:
        """
        Validate LoRA hyperparameters.
        """

        if self.rank <= 0:
            raise ValueError("LoRA rank must be greater than zero.")

        if self.alpha <= 0:
            raise ValueError("LoRA alpha must be greater than zero.")

        if not 0.0 <= self.dropout <= 1.0:
            raise ValueError("LoRA dropout must be between 0.0 and 1.0.")

    def to_dict(self) -> Dict[str, object]:
        """
        Export configuration as dictionary.

        Useful for:
        - MLflow
        - Weights & Biases
        - Training metadata
        """

        config = asdict(self)

        config["task_type"] = str(self.task_type)

        return config


def build_lora_config(
    params: Optional[LoRAHyperParameters] = None,
) -> LoraConfig:
    """
    Build PEFT LoRAConfig object.

    Args:
        params:
            Optional custom LoRA hyperparameters.

    Returns:
        LoraConfig instance.
    """

    params = params or LoRAHyperParameters()

    params.validate()

    return LoraConfig(
        r=params.rank,
        lora_alpha=params.alpha,
        lora_dropout=params.dropout,
        bias=params.bias,
        task_type=params.task_type,
        target_modules=list(params.target_modules),
        modules_to_save=(
            list(params.modules_to_save) if params.modules_to_save else None
        ),
        inference_mode=params.inference_mode,
        use_rslora=params.use_rslora,
        use_dora=params.use_dora,
        init_lora_weights=params.init_lora_weights,
    )


DEFAULT_LORA_HYPERPARAMETERS = LoRAHyperParameters()

DEFAULT_LORA_CONFIG = build_lora_config(DEFAULT_LORA_HYPERPARAMETERS)
