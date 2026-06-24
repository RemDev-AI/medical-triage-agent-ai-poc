# medical-triage-agent-ai-poc/backend/app/training/colab/colab_gpu_detector.py

"""
Google Colab GPU Detection Utilities

Supported GPUs
--------------
- NVIDIA T4
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

Used by:
- backend/app/training/sft/train_sft.py
- backend/app/training/dpo/train_dpo.py
- backend/app/training/evaluation/clinical_eval_runner.py
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
    L4 = "L4"
    V100 = "V100"
    A100 = "A100"
    UNKNOWN = "UNKNOWN"
    CPU = "CPU"


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
        GPUType.L4: 2,
        GPUType.V100: 2,
        GPUType.A100: 4,
    }

    return recommendations.get(gpu_type, 1)


def get_recommended_quantization(
    gpu_type: GPUType,
) -> str:
    """
    Quantization recommendation.
    """

    recommendations = {
        GPUType.T4: "4bit",
        GPUType.L4: "4bit",
        GPUType.V100: "8bit",
        GPUType.A100: "none",
    }

    return recommendations.get(gpu_type, "4bit")


def is_bf16_recommended(
    gpu_type: GPUType,
) -> bool:
    """
    BF16 recommendation.
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
        "Quantization: %s",
        info.quantization,
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
    }


def is_colab_supported_gpu() -> bool:
    """
    Check if GPU belongs to validated list.
    """

    return detect_gpu_type() in {
        GPUType.T4,
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
