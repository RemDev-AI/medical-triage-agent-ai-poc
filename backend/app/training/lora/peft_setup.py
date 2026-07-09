# medical-triage-agent-ai-poc/backend/app/training/lora/peft_setup.py

"""
PEFT integration utilities.

This module handles:
- LoRA adapter injection
- trainable parameter reporting
- GPU memory monitoring
- model preparation for k-bit training

Étape 1 (audit OOM DPO) :
- prepare_model_for_kbit_training() n'est appelée QUE si le modèle est
  effectivement quantifié (4/8 bits). Sur un modèle FP16 complet, cette
  fonction est un no-op trompeur (elle suppose un modèle déjà quantifié :
  cf. doc PEFT) et ne doit pas être invoquée.
- gradient_checkpointing n'est plus ré-activé ici par défaut, car
  TrainingModelLoader.apply_gradient_checkpointing() le fait déjà en
  amont avec use_reentrant=False (requis pour Qwen3, bug #4). Le
  ré-activer ici sans ce kwarg écraserait ce réglage.
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


def is_model_quantized(model: AutoModelForCausalLM) -> bool:
    """
    Détecte si le modèle a été chargé avec une quantification
    (BitsAndBytesConfig 4 bits ou 8 bits), qu'il vienne de
    transformers natif ou d'Unsloth.

    transformers pose `is_loaded_in_4bit` / `is_loaded_in_8bit` sur le
    modèle quand un `quantization_config` a été fourni à
    `from_pretrained()`. Unsloth pose également ces attributs sur les
    modèles qu'il charge en 4 bits.
    """
    return bool(
        getattr(model, "is_loaded_in_4bit", False)
        or getattr(model, "is_loaded_in_8bit", False)
        or getattr(getattr(model, "config", None), "quantization_config", None)
        is not None
    )


def prepare_model_for_lora(
    model: AutoModelForCausalLM,
    gradient_checkpointing: bool = True,
    already_gradient_checkpointed: bool = False,
) -> AutoModelForCausalLM:
    """
    Prepare model for LoRA fine-tuning.

    Includes:
    - k-bit preparation (uniquement si le modèle est quantifié)
    - gradient checkpointing (uniquement si pas déjà activé en amont)
    - input gradients

    Args:
        model:
            Base model.

        gradient_checkpointing:
            Enable gradient checkpointing.

        already_gradient_checkpointed:
            True si TrainingModelLoader a déjà activé le gradient
            checkpointing (avec use_reentrant=False) avant l'appel à
            setup_peft_model(). Dans ce cas, on ne le ré-active pas ici
            pour ne pas écraser ce réglage.

    Returns:
        Prepared model.
    """

    quantized = is_model_quantized(model)

    if quantized:
        logger.info(
            "Modèle quantifié détecté — preparing for k-bit training..."
        )  # noqa: E501
        model = prepare_model_for_kbit_training(
            model,
            use_gradient_checkpointing=gradient_checkpointing
            and not already_gradient_checkpointed,
        )
    else:
        logger.info(
            "Modèle non quantifié (FP16/BF16/FP32) — "
            "prepare_model_for_kbit_training() ignorée."
        )
        if hasattr(model, "config"):
            model.config.use_cache = False

    if (
        gradient_checkpointing and not already_gradient_checkpointed and not quantized
    ):  # noqa: E501
        # Le chemin quantifié gère déjà le gradient checkpointing via
        # prepare_model_for_kbit_training(use_gradient_checkpointing=...).
        logger.info(
            "Enabling gradient checkpointing (use_reentrant=False)..."
        )  # noqa: E501
        model.gradient_checkpointing_enable(
            gradient_checkpointing_kwargs={"use_reentrant": False}
        )

    if hasattr(model, "enable_input_require_grads"):
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
        use_rslora=lora_section.get(
            "use_rslora", _DEFAULT_LORA_PARAMS.use_rslora
        ),  # noqa : E501
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

    # --- Garde-fou dtype poids maîtres (corrige "Option A / audit DPO
    # 2026-07-03") ---
    # PEFT peut, selon la version, caster certains paramètres
    # entraînables (notamment via modules_to_save, ex. lm_head) en
    # bfloat16, indépendamment du dtype du modèle de base chargé en
    # amont (torch.float16 sur T4).
    #
    # L'ancienne version de ce garde-fou recastait vers torch.float16,
    # ce qui est incorrect : le GradScaler utilisé en mixed-precision
    # fp16 (fp16=True côté DPOConfig) exige que les POIDS MAÎTRES des
    # paramètres entraînables restent en float32 — seul l'autocast doit
    # produire du fp16 pendant le forward/backward. Un paramètre
    # entraînable dont le .dtype brut est déjà torch.float16 fait
    # échouer `GradScaler._unscale_grads_` (allow_fp16=False) :
    # "ValueError: Attempting to unscale FP16 gradients."
    #
    # On recast donc tout résidu bfloat16 (et fp16, par cohérence) vers
    # float32, seul dtype de stockage stable avec un GradScaler fp16.
    forced_params = []
    for name, param in peft_model.named_parameters():
        if param.requires_grad and param.dtype in (
            torch.bfloat16,
            torch.float16,
        ):
            param.data = param.data.to(torch.float32)
            forced_params.append(name)

    if forced_params:
        logger.warning(
            "Garde-fou dtype : %d paramètre(s) entraînable(s) forcé(s) "
            "vers float32 (poids maîtres) : %s",
            len(forced_params),
            forced_params,
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

    ratio = 100 * trainable_params / total_params if total_params > 0 else 0.0

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
    config: Optional[Dict] = None,  # ← FIX BUG #2 : paramètre ajouté
    already_gradient_checkpointed: bool = True,  # ← étape 1 : évite le double GC  # noqa : E501
) -> PeftModel:
    """
    Full PEFT setup pipeline.

    Steps:
    - prepare model (k-bit prep si quantifié, sinon simple use_cache=False)
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

        already_gradient_checkpointed:
            True (défaut) si l'appelant (TrainingModelLoader) a déjà
            activé le gradient checkpointing avec use_reentrant=False
            avant d'appeler setup_peft_model(). Passez False si vous
            appelez setup_peft_model() directement sur un modèle brut.

    Returns:
        PEFT model.
    """

    total_params_before = sum(param.numel() for param in model.parameters())

    logger.info(
        "Model parameters before PEFT setup: %s",
        total_params_before,
    )

    model = prepare_model_for_lora(
        model,
        already_gradient_checkpointed=already_gradient_checkpointed,
    )

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
