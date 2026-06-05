# medical-triage-agent-ai-poc/backend/app/deployment/modal/modal_gpu_config.py

"""
modal_gpu_config.py

Gestion centralisée des ressources GPU Modal.

Support :

- NVIDIA A10G
- NVIDIA A100
- NVIDIA H100

Utilisé par :

- modal_deploy.py
- modal_endpoint.py

Objectifs :

- standardisation infra GPU ;
- reproductibilité MLOps ;
- sélection dynamique GPU ;
- optimisation coûts/performance.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum


# =========================================================
# GPU TYPES
# =========================================================


class ModalGPUType(str, Enum):
    """
    Types GPU supportés.
    """

    A10G = "A10G"
    A100 = "A100"
    H100 = "H100"


# =========================================================
# CONFIG MODEL
# =========================================================


@dataclass(frozen=True)
class GPUConfiguration:
    """
    Configuration GPU.
    """

    gpu_name: str
    memory_gb: int
    max_batch_size: int
    recommended_for: str


# =========================================================
# GPU PROFILES
# =========================================================


GPU_CONFIGS: dict[ModalGPUType, GPUConfiguration] = {
    ModalGPUType.A10G: GPUConfiguration(
        gpu_name="A10G",
        memory_gb=24,
        max_batch_size=8,
        recommended_for=(
            "Qwen3-1.7B + LoRA "
            "Inference Production"
        ),
    ),
    ModalGPUType.A100: GPUConfiguration(
        gpu_name="A100",
        memory_gb=80,
        max_batch_size=32,
        recommended_for=(
            "SFT Training "
            "+ DPO Training"
        ),
    ),
    ModalGPUType.H100: GPUConfiguration(
        gpu_name="H100",
        memory_gb=80,
        max_batch_size=64,
        recommended_for=(
            "High Throughput vLLM "
            "Production"
        ),
    ),
}


# =========================================================
# DEFAULT CONFIG
# =========================================================


DEFAULT_GPU = ModalGPUType.A10G


# =========================================================
# ACCESSORS
# =========================================================


def get_gpu_config(
    gpu_type: ModalGPUType = DEFAULT_GPU,
) -> GPUConfiguration:
    """
    Retourne la configuration GPU.
    """

    return GPU_CONFIGS[gpu_type]


def get_training_gpu() -> GPUConfiguration:
    """
    GPU recommandé entraînement.
    """

    return GPU_CONFIGS[
        ModalGPUType.A100
    ]


def get_inference_gpu() -> GPUConfiguration:
    """
    GPU recommandé inférence.
    """

    return GPU_CONFIGS[
        ModalGPUType.A10G
    ]


def get_high_performance_gpu() -> GPUConfiguration:
    """
    GPU recommandé forte charge.
    """

    return GPU_CONFIGS[
        ModalGPUType.H100
    ]


# =========================================================
# DISPLAY
# =========================================================


def describe_gpu(
    gpu_type: ModalGPUType,
) -> str:
    """
    Description lisible.
    """

    config = get_gpu_config(
        gpu_type
    )

    return (
        f"{config.gpu_name} | "
        f"{config.memory_gb} GB | "
        f"Batch={config.max_batch_size} | "
        f"{config.recommended_for}"
    )


# =========================================================
# MAIN
# =========================================================


if __name__ == "__main__":

    for gpu in ModalGPUType:

        print(
            describe_gpu(gpu)
        )
