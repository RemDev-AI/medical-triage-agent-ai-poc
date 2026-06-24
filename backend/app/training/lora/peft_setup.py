# medical-triage-agent-ai-poc/backend/app/training/lora/peft_setup.py

"""
PEFT integration utilities.

This module handles:
- LoRA adapter injection
- trainable parameter reporting
- GPU memory monitoring
- model preparation for k-bit training
"""

import logging
from typing import Tuple

import torch
from peft import (
    get_peft_model,
    prepare_model_for_kbit_training,
)
from transformers import AutoModelForCausalLM

from backend.app.training.lora.lora_config import DEFAULT_LORA_CONFIG

logger = logging.getLogger(__name__)


def prepare_model_for_lora(
    model: AutoModelForCausalLM,
    gradient_checkpointing: bool = True,
):
    """
    Prepare model for LoRA fine-tuning.

    Includes:
    - k-bit preparation
    - gradient checkpointing
    - input gradients

    Args:
        model:
            Base model.

        gradient_checkpointing:
            Enable gradient checkpointing.

    Returns:
        Prepared model.
    """

    logger.info("Preparing model for k-bit training...")

    model = prepare_model_for_kbit_training(model)

    if gradient_checkpointing:
        logger.info("Enabling gradient checkpointing...")
        model.gradient_checkpointing_enable()

    model.enable_input_require_grads()

    return model


def inject_lora_adapters(
    model: AutoModelForCausalLM,
):
    """
    Inject LoRA adapters into model.

    Args:
        model:
            Prepared model.

    Returns:
        PEFT wrapped model.
    """

    logger.info("Injecting LoRA adapters...")

    peft_model = get_peft_model(
        model,
        DEFAULT_LORA_CONFIG,
    )

    logger.info("LoRA adapters successfully injected.")

    return peft_model


def print_trainable_parameters(
    model,
) -> Tuple[int, int]:
    """
    Print trainable parameter statistics.

    Args:
        model:
            PEFT model.

    Returns:
        Tuple containing:
        - trainable parameters
        - total parameters
    """

    trainable_params = 0
    total_params = 0

    for _, param in model.named_parameters():
        total_params += param.numel()

        if param.requires_grad:
            trainable_params += param.numel()

    ratio = 100 * trainable_params / total_params

    logger.info(
        "Trainable params: %s | Total params: %s | Trainable ratio: %.4f%%",
        trainable_params,
        total_params,
        ratio,
    )

    return trainable_params, total_params


def get_gpu_memory_usage() -> dict:
    """
    Retrieve GPU memory usage.

    Returns:
        Dictionary containing GPU memory stats.
    """

    if not torch.cuda.is_available():
        return {
            "cuda_available": False,
        }

    device = torch.cuda.current_device()

    allocated = torch.cuda.memory_allocated(device) / 1024**3
    reserved = torch.cuda.memory_reserved(device) / 1024**3
    max_reserved = torch.cuda.max_memory_reserved(device) / 1024**3

    return {
        "cuda_available": True,
        "device": torch.cuda.get_device_name(device),
        "allocated_gb": round(allocated, 2),
        "reserved_gb": round(reserved, 2),
        "max_reserved_gb": round(max_reserved, 2),
    }


def setup_peft_model(
    model: AutoModelForCausalLM,
):
    """
    Full PEFT setup pipeline.

    Steps:
    - prepare model
    - inject LoRA
    - print trainable params

    Args:
        model:
            Base transformer model.

    Returns:
        PEFT model.
    """

    model = prepare_model_for_lora(model)

    model = inject_lora_adapters(model)

    print_trainable_parameters(model)

    gpu_stats = get_gpu_memory_usage()

    logger.info("GPU stats: %s", gpu_stats)

    return model
