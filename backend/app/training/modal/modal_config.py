# medical-triage-agent-ai-poc/backend/app/training/modal/modal_config.py

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


app = modal.App(APP_NAME)


training_volume = modal.Volume.from_name(
    TRAINING_VOLUME_NAME,
    create_if_missing=True,
)


training_image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install(
        "torch",
        "transformers",
        "datasets",
        "trl",
        "peft",
        "accelerate",
        "bitsandbytes",
        "huggingface_hub",
        "wandb",
        # "mlflow",
        "pyyaml",
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
