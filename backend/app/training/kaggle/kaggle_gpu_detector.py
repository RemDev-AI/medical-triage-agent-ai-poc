# medical-triage-agent-ai-poc/backend/app/training/kaggle/kaggle_gpu_detector.py

# Correctif (audit OOM DPO, étape 4) :
#   - QUANT-1 : get_recommended_quantization() retourne un tri-état
#               "4bit"/"8bit"/"none", mais training_model_loader.py
#               n'implémente QUE le 4 bits (BitsAndBytesConfig via
#               load_in_4bit=True). La recommandation "8bit" pour V100
#               n'est consommable par aucun composant du pipeline.
#               Ajout de should_use_4bit_quantization(), fonction booléenne
#               alignée sur le support réel, destinée à être branchée par
#               kaggle_environment.py sur config["quantization"]["enabled"].
#               get_recommended_quantization() est conservée à titre
#               informatif/logging uniquement — NE PAS l'utiliser pour
#               piloter la config.

"""
Kaggle Notebooks GPU Detection Utilities

Supported GPUs
--------------
- NVIDIA T4 (single or dual T4 accelerator on Kaggle)
- NVIDIA P100
- NVIDIA L4
- NVIDIA V100
- NVIDIA A100

Features
--------
- GPU detection
- VRAM detection
- GPU classification
- LoRA recommendations
- DPO recommendations
- Quantization recommendations
- BF16 capability guidance

Notes on Kaggle specifics
--------------------------
- Kaggle's most common accelerator tiers are "GPU T4 x2" (dual T4,
  16 GB VRAM each) and "GPU P100" (single P100, 16 GB VRAM). L4/A100/V100
  are not standard Kaggle accelerator options at the time of writing, but
  are kept here for forward-compatibility and to preserve a single
  consistent GPU classification surface with the Colab implementation.
- `torch.cuda.get_device_name(0)` only reports the *first* visible GPU.
  On the "GPU T4 x2" tier, this module deliberately only characterizes
  device 0 — multi-GPU orchestration (e.g. via `accelerate` or
  `torch.nn.DataParallel`) is expected to be handled by the training
  scripts themselves, not by this detector.

Used by:
- backend/app/training/sft/train_sft.py
- backend/app/training/dpo/train_dpo.py
- backend/app/training/evaluation/clinical_eval_runner.py
- backend/app/training/kaggle/kaggle_environment.py (pilotage automatique
  de quantization.enabled / torch_dtype à partir de ces recommandations)
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from enum import Enum

import torch

logger = logging.getLogger(__name__)


class GPUType(str, Enum):
    """
    Supported GPU families.
    """

    T4 = "T4"
    P100 = "P100"
    L4 = "L4"
    V100 = "V100"
    A100 = "A100"
    UNKNOWN = "UNKNOWN"
    CPU = "CPU"


# FIX QUANT-1 — GPU sur lesquels la quantification 4 bits est recommandée.
# A100 dispose de suffisamment de VRAM (40/80 Go) pour se passer de
# quantification ; les autres GPU listés bénéficient du 4 bits.
# P100 (16 Go, pas de support Tensor Core BF16) est traité comme T4 :
# la quantification 4 bits est recommandée pour préserver la VRAM.
# UNKNOWN et CPU sont traités prudemment : 4 bits par défaut si CUDA
# disponible (mieux vaut économiser la VRAM sur un GPU non identifié),
# aucune quantification si CPU (non pertinent / non supporté par bnb).
_GPUS_RECOMMENDING_4BIT = {
    GPUType.T4,
    GPUType.P100,
    GPUType.L4,
    GPUType.V100,
    GPUType.UNKNOWN,
}


@dataclass
class GPUInfo:
    """
    GPU metadata.
    """

    gpu_type: GPUType
    gpu_name: str
    total_vram_gb: float

    bf16_recommended: bool
    fp16_recommended: bool

    lora_batch_size: int
    dpo_batch_size: int

    quantization: str
    # FIX QUANT-1 — champ booléen consommable directement par
    # kaggle_environment.py, aligné sur le seul mode réellement implémenté
    # (QLoRA 4 bits via BitsAndBytesConfig).
    quantization_4bit_recommended: bool

    cuda_available: bool


def get_gpu_name() -> str:
    """
    Return CUDA device name.
    """

    if not torch.cuda.is_available():
        return "CPU"

    try:
        return torch.cuda.get_device_name(0)

    except Exception:
        return "Unknown GPU"


def get_total_vram_gb() -> float:
    """
    Return total VRAM in GB.
    """

    if not torch.cuda.is_available():
        return 0.0

    try:
        props = torch.cuda.get_device_properties(0)
        return round(props.total_memory / (1024**3), 2)

    except Exception:
        return 0.0


def detect_gpu_type() -> GPUType:
    """
    Detect GPU family.
    """

    if not torch.cuda.is_available():
        return GPUType.CPU

    gpu_name = get_gpu_name().upper()

    if "A100" in gpu_name:
        return GPUType.A100

    if "L4" in gpu_name:
        return GPUType.L4

    if "V100" in gpu_name:
        return GPUType.V100

    if "P100" in gpu_name:
        return GPUType.P100

    if "T4" in gpu_name:
        return GPUType.T4

    return GPUType.UNKNOWN


def get_recommended_lora_batch_size(
    gpu_type: GPUType,
) -> int:
    """
    Recommended LoRA batch size.
    """

    recommendations = {
        GPUType.T4: 2,
        GPUType.P100: 2,
        GPUType.L4: 4,
        GPUType.V100: 4,
        GPUType.A100: 8,
    }

    return recommendations.get(gpu_type, 1)


def get_recommended_dpo_batch_size(
    gpu_type: GPUType,
) -> int:
    """
    Recommended DPO batch size.
    """

    recommendations = {
        GPUType.T4: 1,
        GPUType.P100: 1,
        GPUType.L4: 2,
        GPUType.V100: 2,
        GPUType.A100: 4,
    }

    return recommendations.get(gpu_type, 1)


def get_recommended_quantization(
    gpu_type: GPUType,
) -> str:
    """
    Quantization recommendation (INFORMATIF / LOGGING UNIQUEMENT).

    ATTENTION : cette fonction peut retourner "8bit", un mode NON
    implémenté par training_model_loader.py (QLoRA 4 bits uniquement,
    via BitsAndBytesConfig(load_in_4bit=True)). Ne pas utiliser cette
    valeur pour piloter config["quantization"] — utiliser
    should_use_4bit_quantization() à la place, qui est garantie
    cohérente avec ce que le pipeline sait réellement charger.
    """

    recommendations = {
        GPUType.T4: "4bit",
        GPUType.P100: "4bit",
        GPUType.L4: "4bit",
        GPUType.V100: "8bit",
        GPUType.A100: "none",
    }

    return recommendations.get(gpu_type, "4bit")


def should_use_4bit_quantization(
    gpu_type: GPUType,
) -> bool:
    """
    FIX QUANT-1 — Recommandation booléenne alignée sur le support réel
    du pipeline (QLoRA 4 bits uniquement).

    Destinée à être branchée directement sur
    config["quantization"]["enabled"] par kaggle_environment.py.
    """

    return gpu_type in _GPUS_RECOMMENDING_4BIT


def is_bf16_recommended(
    gpu_type: GPUType,
) -> bool:
    """
    BF16 recommendation.

    P100 n'a pas de support Tensor Core BF16 (architecture Pascal,
    antérieure à Turing/Ampere) et est donc volontairement exclu, au
    même titre que T4 et V100.
    """

    return gpu_type in {
        GPUType.L4,
        GPUType.A100,
    }


def build_gpu_info() -> GPUInfo:
    """
    Build GPU metadata structure.
    """

    gpu_type = detect_gpu_type()

    bf16 = is_bf16_recommended(gpu_type)

    return GPUInfo(
        gpu_type=gpu_type,
        gpu_name=get_gpu_name(),
        total_vram_gb=get_total_vram_gb(),
        bf16_recommended=bf16,
        fp16_recommended=not bf16,
        lora_batch_size=get_recommended_lora_batch_size(
            gpu_type
        ),
        dpo_batch_size=get_recommended_dpo_batch_size(
            gpu_type
        ),
        quantization=get_recommended_quantization(
            gpu_type
        ),
        quantization_4bit_recommended=should_use_4bit_quantization(
            gpu_type
        ),
        cuda_available=torch.cuda.is_available(),
    )


def log_gpu_info() -> GPUInfo:
    """
    Log GPU information.
    """

    info = build_gpu_info()

    logger.info("========== GPU DETECTION ==========")
    logger.info("CUDA Available: %s", info.cuda_available)
    logger.info("GPU Name: %s", info.gpu_name)
    logger.info("GPU Type: %s", info.gpu_type.value)
    logger.info("VRAM (GB): %.2f", info.total_vram_gb)
    logger.info(
        "BF16 Recommended: %s",
        info.bf16_recommended,
    )
    logger.info(
        "FP16 Recommended: %s",
        info.fp16_recommended,
    )
    logger.info(
        "LoRA Batch Size: %s",
        info.lora_batch_size,
    )
    logger.info(
        "DPO Batch Size: %s",
        info.dpo_batch_size,
    )
    logger.info(
        "Quantization (info) : %s",
        info.quantization,
    )
    logger.info(
        "Quantization 4bit recommended (utilisé par le pipeline): %s",
        info.quantization_4bit_recommended,
    )
    logger.info("===================================")

    return info


def get_training_recommendations() -> dict:
    """
    Return training recommendations.

    Example:
        recommendations =
            get_training_recommendations()
    """

    gpu = build_gpu_info()

    return {
        "gpu_type": gpu.gpu_type.value,
        "gpu_name": gpu.gpu_name,
        "vram_gb": gpu.total_vram_gb,
        "bf16": gpu.bf16_recommended,
        "fp16": gpu.fp16_recommended,
        "lora_batch_size": gpu.lora_batch_size,
        "dpo_batch_size": gpu.dpo_batch_size,
        "quantization": gpu.quantization,
        "quantization_4bit_recommended": gpu.quantization_4bit_recommended,
    }


def is_kaggle_supported_gpu() -> bool:
    """
    Check if GPU belongs to validated list.
    """

    return detect_gpu_type() in {
        GPUType.T4,
        GPUType.P100,
        GPUType.L4,
        GPUType.V100,
        GPUType.A100,
    }


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    gpu_info = log_gpu_info()

    print()
    print("GPU Summary")
    print("-----------")
    print(gpu_info)
