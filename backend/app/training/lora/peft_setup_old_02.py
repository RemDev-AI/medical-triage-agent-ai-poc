# medical-triage-agent-ai-poc/backend/app/training/lora/peft_setup.py

"""
PEFT integration utilities.

This module handles:
- LoRA adapter injection
- trainable parameter reporting
- GPU memory monitoring
- model preparation for k-bit training
"""

from __future__ import annotations

import logging
from typing import Dict, Optional, Tuple

import torch
from peft import (
    LoraConfig,
    PeftModel,
    TaskType,
    get_peft_model,
    prepare_model_for_kbit_training,
)
from transformers import AutoModelForCausalLM

from backend.app.training.lora.lora_config import (
    DEFAULT_LORA_CONFIG,
    LoRAHyperParameters,
    build_lora_config,
)

logger = logging.getLogger(__name__)

# Instance de référence pour les valeurs par défaut
# (dataclass : les defaults ne sont pas des attributs de classe)
_DEFAULT_LORA_PARAMS = LoRAHyperParameters()


def prepare_model_for_lora(
    model: AutoModelForCausalLM,
    gradient_checkpointing: bool = True,
) -> AutoModelForCausalLM:
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

    if hasattr(model, "config"):
        model.config.use_cache = False

    if gradient_checkpointing:
        logger.info("Enabling gradient checkpointing...")
        model.gradient_checkpointing_enable()

    model.enable_input_require_grads()

    return model


def build_lora_config_from_yaml(config: Dict) -> LoraConfig:
    """
    Build a LoraConfig from the YAML training config dict.

    Reads from config["lora"] section. Falls back to
    LoRAHyperParameters defaults for any missing key.

    Args:
        config:
            Full training config dict (loaded from YAML).

    Returns:
        LoraConfig instance.
    """
    lora_section = config.get("lora", {})

    # _DEFAULT_LORA_PARAMS est une instance : accès aux defaults garanti
    params = LoRAHyperParameters(
        rank=lora_section.get("r", _DEFAULT_LORA_PARAMS.rank),
        alpha=lora_section.get("lora_alpha", _DEFAULT_LORA_PARAMS.alpha),
        dropout=lora_section.get("lora_dropout", _DEFAULT_LORA_PARAMS.dropout),
        bias=lora_section.get("bias", _DEFAULT_LORA_PARAMS.bias),
        task_type=TaskType[lora_section.get("task_type", "CAUSAL_LM")],
        target_modules=tuple(
            lora_section.get(
                "target_modules",
                list(_DEFAULT_LORA_PARAMS.target_modules),
            )
        ),
        modules_to_save=(
            tuple(lora_section["modules_to_save"])
            if "modules_to_save" in lora_section
            else _DEFAULT_LORA_PARAMS.modules_to_save
        ),
        inference_mode=lora_section.get(
            "inference_mode", _DEFAULT_LORA_PARAMS.inference_mode
        ),
        use_rslora=lora_section.get("use_rslora", _DEFAULT_LORA_PARAMS.use_rslora),  # noqa : E501
        use_dora=lora_section.get("use_dora", _DEFAULT_LORA_PARAMS.use_dora),
        init_lora_weights=lora_section.get(
            "init_lora_weights", _DEFAULT_LORA_PARAMS.init_lora_weights
        ),
    )

    params.validate()

    logger.info(
        "LoRA config built from YAML: rank=%s, alpha=%s, dropout=%s, "
        "target_modules=%s",
        params.rank,
        params.alpha,
        params.dropout,
        params.target_modules,
    )

    return build_lora_config(params)


def inject_lora_adapters(
    model: AutoModelForCausalLM,
    lora_config: Optional[LoraConfig] = None,
) -> PeftModel:
    """
    Inject LoRA adapters into model.

    Args:
        model:
            Prepared model.

        lora_config:
            LoraConfig to use. Falls back to DEFAULT_LORA_CONFIG
            if not provided.

    Returns:
        PEFT wrapped model.
    """

    effective_config = lora_config or DEFAULT_LORA_CONFIG

    if effective_config is None:
        raise ValueError(
            "No LoraConfig provided and DEFAULT_LORA_CONFIG is not configured."
        )

    logger.info("Injecting LoRA adapters...")

    peft_model = get_peft_model(
        model,
        effective_config,
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

    ratio = (
        100 * trainable_params / total_params
        if total_params > 0
        else 0.0
    )

    logger.info(
        "Trainable params: %s | Total params: %s | Trainable ratio: %.4f%%",
        trainable_params,
        total_params,
        ratio,
    )

    return trainable_params, total_params


def get_gpu_memory_usage() -> Dict[str, object]:
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
    max_allocated = torch.cuda.max_memory_allocated(device) / 1024**3
    max_reserved = torch.cuda.max_memory_reserved(device) / 1024**3

    return {
        "cuda_available": True,
        "device": torch.cuda.get_device_name(device),
        "device_count": torch.cuda.device_count(),
        "allocated_gb": round(allocated, 2),
        "reserved_gb": round(reserved, 2),
        "max_allocated_gb": round(max_allocated, 2),
        "max_reserved_gb": round(max_reserved, 2),
    }


def setup_peft_model(
    model: AutoModelForCausalLM,
    config: Optional[Dict] = None,      # ← FIX BUG #2 : paramètre ajouté
) -> PeftModel:
    """
    Full PEFT setup pipeline.

    Steps:
    - prepare model
    - build LoraConfig (depuis YAML si config fourni, sinon DEFAULT)
    - inject LoRA
    - print trainable params
    - log GPU statistics

    Args:
        model:
            Base transformer model.

        config:
            Full training config dict (loaded from YAML).
            Si fourni, les hyperparamètres LoRA sont lus depuis
            config["lora"]. Si None, DEFAULT_LORA_CONFIG est utilisé.

    Returns:
        PEFT model.
    """

    total_params_before = sum(
        param.numel()
        for param in model.parameters()
    )

    logger.info(
        "Model parameters before PEFT setup: %s",
        total_params_before,
    )

    model = prepare_model_for_lora(model)

    # FIX BUG #2 — construire la LoraConfig depuis le YAML si disponible
    if config is not None:
        lora_config = build_lora_config_from_yaml(config)
    else:
        lora_config = DEFAULT_LORA_CONFIG
        logger.warning(
            "No config provided to setup_peft_model — "
            "using DEFAULT_LORA_CONFIG (YAML ignoré)."
        )

    peft_model = inject_lora_adapters(model, lora_config=lora_config)

    print_trainable_parameters(peft_model)

    gpu_stats = get_gpu_memory_usage()

    logger.info("GPU stats: %s", gpu_stats)

    return peft_model
