# medical-triage-agent-ai-poc/backend/app/training/colab/colab_environment.py

"""
Google Colab Environment Detection Utilities

Features
--------
- Detect Google Colab runtime
- Detect CUDA availability
- Detect BF16 support
- Detect FP16 support
- Expose unified training environment settings
- Centralized BF16 logic via colab_gpu_detector.py

Used by:
- backend/app/training/sft/train_sft.py
- backend/app/training/dpo/train_dpo.py
- backend/app/training/evaluation/clinical_eval_runner.py
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import torch

from backend.app.training.colab.colab_gpu_detector import (
    build_gpu_info,
)

logger = logging.getLogger(__name__)


@dataclass
class ColabEnvironment:
    """
    Runtime environment description.
    """

    is_colab: bool
    cuda_available: bool
    device: str

    bf16_supported: bool
    fp16_supported: bool

    torch_version: str
    cuda_version: str | None

    gpu_name: str | None


def is_running_in_colab() -> bool:
    """
    Detect whether execution occurs inside Google Colab.
    """

    try:
        import google.colab  # noqa: F401

        return True

    except Exception:
        return False


def get_gpu_name() -> str | None:
    """
    Return GPU name if CUDA is available.
    """

    if not torch.cuda.is_available():
        return None

    try:
        return torch.cuda.get_device_name(0)

    except Exception:
        return None


def is_bf16_supported() -> bool:
    """
    Detect BF16 capability using validated
    GPU profiles from colab_gpu_detector.py.

    Supported BF16 GPUs:
    - NVIDIA L4
    - NVIDIA A100
    """

    if not torch.cuda.is_available():
        return False

    try:
        gpu_info = build_gpu_info()
        return gpu_info.bf16_recommended

    except Exception:
        return False


def is_fp16_supported() -> bool:
    """
    FP16 fallback support.

    Any CUDA-enabled GPU used in Colab can run FP16.
    """

    return torch.cuda.is_available()


def get_training_dtype() -> torch.dtype:
    """
    Return optimal training dtype.

    Priority:
    BF16 -> FP16 -> FP32
    """

    if is_bf16_supported():
        return torch.bfloat16

    if is_fp16_supported():
        return torch.float16

    return torch.float32


def get_device() -> str:
    """
    Resolve runtime device.
    """

    return "cuda" if torch.cuda.is_available() else "cpu"


def detect_environment() -> ColabEnvironment:
    """
    Build runtime environment descriptor.
    """

    cuda_version = (
        torch.version.cuda
        if torch.cuda.is_available()
        else None
    )

    return ColabEnvironment(
        is_colab=is_running_in_colab(),
        cuda_available=torch.cuda.is_available(),
        device=get_device(),
        bf16_supported=is_bf16_supported(),
        fp16_supported=is_fp16_supported(),
        torch_version=torch.__version__,
        cuda_version=cuda_version,
        gpu_name=get_gpu_name(),
    )


def log_environment_info() -> ColabEnvironment:
    """
    Log runtime information.
    """

    env = detect_environment()

    logger.info("========== TRAINING ENVIRONMENT ==========")
    logger.info("Google Colab: %s", env.is_colab)
    logger.info("Device: %s", env.device)
    logger.info("CUDA Available: %s", env.cuda_available)
    logger.info("GPU: %s", env.gpu_name)
    logger.info("BF16 Supported: %s", env.bf16_supported)
    logger.info("FP16 Supported: %s", env.fp16_supported)
    logger.info("Torch Version: %s", env.torch_version)
    logger.info("CUDA Version: %s", env.cuda_version)
    logger.info("Training DType: %s", get_training_dtype())
    logger.info("==========================================")

    return env


def get_training_arguments_precision() -> dict[str, bool]:
    """
    Return Hugging Face Trainer precision arguments.

    Example:
        trainer_args.update(
            get_training_arguments_precision()
        )
    """

    bf16 = is_bf16_supported()

    return {
        "bf16": bf16,
        "fp16": (
            torch.cuda.is_available()
            and not bf16
        ),
    }


def apply_precision_arguments(
    training_args: dict,
) -> dict:
    """
    Inject precision configuration into a
    TrainingArguments configuration dictionary.

    Example:
        args = {
            "per_device_train_batch_size": 4,
        }

        args = apply_precision_arguments(args)
    """

    training_args.update(
        get_training_arguments_precision()
    )

    return training_args


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    environment = log_environment_info()

    print()
    print("Environment Summary")
    print("-------------------")
    print(environment)

    print()
    print("Trainer Precision")
    print("-----------------")
    print(get_training_arguments_precision())
