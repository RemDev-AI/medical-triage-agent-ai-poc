# medical-triage-agent-ai-poc/backend/app/training/lora/lora_config.py

"""
Centralized LoRA configuration.

This module defines:
- LoRA hyperparameters
- target modules
- quantization compatibility
- reproducible configuration objects
"""

from dataclasses import dataclass, field
from typing import List, Optional

from peft import LoraConfig, TaskType


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

    target_modules: List[str] = field(
        default_factory=lambda: [
            "q_proj",
            "k_proj",
            "v_proj",
            "o_proj",
            "gate_proj",
            "up_proj",
            "down_proj",
        ]
    )

    modules_to_save: Optional[List[str]] = field(
        default_factory=lambda: [
            "lm_head",
        ]
    )

    inference_mode: bool = False

    use_rslora: bool = False
    use_dora: bool = False

    init_lora_weights: str = "gaussian"


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

    return LoraConfig(
        r=params.rank,
        lora_alpha=params.alpha,
        lora_dropout=params.dropout,
        bias=params.bias,
        task_type=params.task_type,
        target_modules=params.target_modules,
        modules_to_save=params.modules_to_save,
        inference_mode=params.inference_mode,
        use_rslora=params.use_rslora,
        use_dora=params.use_dora,
        init_lora_weights=params.init_lora_weights,
    )


DEFAULT_LORA_CONFIG = build_lora_config()
