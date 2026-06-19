# medical-triage-agent-ai-poc/backend/app/training/modal/modal_config.py
# 
"""
Central Modal configuration.

Responsibilities:
- GPU provisioning
- Hugging Face authentication
- Shared Modal image
- Training volumes
- Environment variables
"""

from __future__ import annotations

from dataclasses import dataclass

import modal


APP_NAME = "medical-triage-agent-ai-poc-training"

HF_SECRET_NAME = "huggingface-secret"
WANDB_SECRET_NAME = "wandb-secret"
# MLFLOW_SECRET_NAME = "mlflow-secret"

TRAINING_VOLUME_NAME = (
    "medical-triage-agent-ai-poc-training-volume"
)

CHECKPOINT_DIR = "/training/checkpoints"
DATASET_DIR = "/training/datasets"

# HF_DATASET_REPO = (
#     "medical-triage-agent-ai-poc-datasets"
# )

# HF_MODEL_REPO = (
#     "medical-triage-agent-ai-poc-models"
# )

HF_NAMESPACE = "RemDev-AI"

HF_DATASET_REPO = (
    f"{HF_NAMESPACE}/medical-triage-agent-ai-poc-datasets"
)

HF_MODEL_REPO = (
    f"{HF_NAMESPACE}/medical-triage-agent-ai-poc-models"
)

HF_DATASET_REVISION = "main"

HF_MODEL_REVISION = "main"

HF_DATASET_CONFIG_SFT = "sft"
HF_DATASET_CONFIG_DPO = "dpo"

SUPPORTED_DATASET_CONFIGS = {
    HF_DATASET_CONFIG_SFT,
    HF_DATASET_CONFIG_DPO,
}

DEFAULT_GPU = "A100-40GB"

DEFAULT_TIMEOUT = 60 * 60 * 24

DEFAULT_CPU = 8

DEFAULT_MEMORY_MB = 65536


@dataclass(frozen=True)
class ModalTrainingConfig:
    """
    Shared training configuration.
    """

    gpu_type: str = DEFAULT_GPU

    timeout: int = DEFAULT_TIMEOUT

    cpu: int = DEFAULT_CPU

    memory_mb: int = DEFAULT_MEMORY_MB

    dataset_repo: str = HF_DATASET_REPO

    model_repo: str = HF_MODEL_REPO

    checkpoint_dir: str = CHECKPOINT_DIR

    dataset_dir: str = DATASET_DIR


config = ModalTrainingConfig()


def validate_modal_config() -> None:
    """
    Validate shared Modal configuration.
    """

    if not HF_DATASET_REPO.startswith(
        f"{HF_NAMESPACE}/"
    ):
        raise ValueError(
            f"Invalid HF_DATASET_REPO: "
            f"{HF_DATASET_REPO}"
        )

    if not HF_MODEL_REPO.startswith(
        f"{HF_NAMESPACE}/"
    ):
        raise ValueError(
            f"Invalid HF_MODEL_REPO: "
            f"{HF_MODEL_REPO}"
        )


validate_modal_config()

app = modal.App(APP_NAME)


training_volume = modal.Volume.from_name(
    TRAINING_VOLUME_NAME,
    create_if_missing=True,
)

from pathlib import Path  # noqa: E402

PROJECT_ROOT = Path(__file__).resolve().parents[4]

training_image = (
    modal.Image.debian_slim(
        python_version="3.11"
    )
    .pip_install(
        "torch>=2.4.0",
        "transformers>=4.52.0",
        "datasets>=3.0.0",
        "trl>=0.9.6",
        "peft>=0.12.0",
        "accelerate>=1.0.1",
        "bitsandbytes>=0.45.0",
        "huggingface_hub>=0.34.0",
        "sentencepiece>=0.2.0",
        "wandb>=0.19.0",
        # "mlflow",
        "pyyaml>=6.0.2",
    )
    .env(
        {
            "PYTHONPATH": "/root",
            "HF_HUB_ENABLE_HF_TRANSFER": "1",
            "TOKENIZERS_PARALLELISM": "false",
        }
    )
    .add_local_dir(
        local_path=str(PROJECT_ROOT / "backend"),
        remote_path="/root/backend",
        copy=True,
    )
)

hf_secret = modal.Secret.from_name(
    HF_SECRET_NAME,
    required_keys=["HF_TOKEN_06"],
)

wandb_secret = modal.Secret.from_name(
    WANDB_SECRET_NAME,
    required_keys=["WANDB_API_KEY"],
)

# mlflow_secret = modal.Secret.from_name(
#     MLFLOW_SECRET_NAME,
#     required_keys=[
#         "MLFLOW_TRACKING_URI",
#     ],
# )
