# medical-triage-agent-ai-poc/backend/app/training/kaggle/kaggle_environment.py

# Correctif (audit OOM DPO, étape 4) :
#   - QUANT-2 : ajout de is_4bit_quantization_recommended() et
#               resolve_quantization_settings(), qui exploitent
#               should_use_4bit_quantization() (kaggle_gpu_detector.py,
#               QUANT-1) pour recommander/avertir sur la config
#               quantization, SANS écraser silencieusement un choix
#               explicite déjà présent dans le YAML (principe : la config
#               explicite de l'utilisateur prime toujours sur la
#               recommandation automatique).

"""
Kaggle Notebooks Environment Detection Utilities

Features
--------
- Detect Kaggle Notebooks runtime
- Detect CUDA availability
- Detect BF16 support
- Detect FP16 support
- Resolve runtime device and dtype
- Build Hugging Face TrainingArguments precision policy
- Recommend / validate quantization settings against detected hardware

Used by:
- backend/app/training/sft/train_sft.py
- backend/app/training/dpo/train_dpo.py
- backend/app/training/evaluation/clinical_eval_runner.py

Notes on Kaggle specifics
--------------------------
- Kaggle Notebooks typically expose either a single T4/P100 GPU or
  a dual-T4 configuration, and more recently L4 GPUs on some
  accelerator tiers. Kaggle does not expose a Colab-style A100/L4-only
  "Pro" runtime, so the T4-related BF16 fallback logic below is
  especially relevant.
- Detection relies on the `KAGGLE_KERNEL_RUN_TYPE` environment variable
  (set by Kaggle for every Notebook session), with `/kaggle/` filesystem
  presence as a secondary signal.
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass
from typing import Any

import torch

from backend.app.training.kaggle.kaggle_gpu_detector import (
    build_gpu_info,
    detect_gpu_type,
    should_use_4bit_quantization,
)

logger = logging.getLogger(__name__)


@dataclass(slots=True)
class KaggleEnvironment:
    """
    Runtime environment description.
    """

    is_kaggle: bool
    cuda_available: bool
    device: str

    bf16_supported: bool
    fp16_supported: bool

    torch_version: str
    cuda_version: str | None

    gpu_name: str | None


def is_running_in_kaggle() -> bool:
    """
    Detect whether execution occurs inside a Kaggle Notebook.

    Kaggle sets the `KAGGLE_KERNEL_RUN_TYPE` environment variable
    (e.g. "Interactive" or "Batch") for every Notebook session. As a
    secondary signal, the `/kaggle` working directory is checked, since
    it is only present on Kaggle's infrastructure.
    """

    if os.environ.get("KAGGLE_KERNEL_RUN_TYPE") is not None:
        return True

    if os.environ.get("KAGGLE_URL_BASE") is not None:
        return True

    try:
        return os.path.isdir("/kaggle")

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
    GPU profiles.
    """

    if not torch.cuda.is_available():
        return False

    try:
        gpu_info = build_gpu_info()
        return gpu_info.bf16_recommended

    except Exception:
        logger.exception("Unable to determine BF16 support.")
        return False


def is_fp16_supported() -> bool:
    """
    Detect FP16 capability.
    """

    return torch.cuda.is_available()


def get_training_dtype() -> torch.dtype:
    """
    Resolve optimal torch dtype.

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


# ---------------------------------------------------------------------------
# FIX QUANT-2 — Recommandation / validation de la quantification
# ---------------------------------------------------------------------------
def is_4bit_quantization_recommended() -> bool:
    """
    Indique si la quantification 4 bits (QLoRA) est recommandée pour le
    GPU détecté au runtime, en s'appuyant sur
    kaggle_gpu_detector.should_use_4bit_quantization() — seul mode
    réellement implémenté par training_model_loader.py.

    Retourne False si aucun GPU CUDA n'est détecté (bitsandbytes 4 bits
    nécessite CUDA).
    """

    if not torch.cuda.is_available():
        return False

    return should_use_4bit_quantization(detect_gpu_type())


def resolve_quantization_settings(
    config: dict[str, Any],
) -> dict[str, Any]:
    """
    Recommande/valide config["quantization"] par rapport au GPU détecté,
    SANS écraser un choix explicite déjà présent dans le YAML.

    Comportement :
    - Si "quantization" ou "quantization.enabled" est absent du YAML :
      applique la recommandation matérielle (is_4bit_quantization_recommended()).  # noqa: E501
    - Si "quantization.enabled" est explicitement défini dans le YAML :
      conservé tel quel — c'est un choix assumé de l'utilisateur (ex. le
      run de validation DPO de ce projet force enabled=true même sur
      matériel où ce ne serait pas strictement nécessaire). Un warning
      est loggué si ce choix diverge de la recommandation matérielle,
      à titre informatif uniquement.

    Args:
        config:
            Config d'entraînement complète (dict chargé depuis le YAML).
            Modifié en place ET retourné pour confort d'appel.

    Returns:
        Le dict config, avec config["quantization"] garanti présent et
        contenant au moins la clé "enabled".
    """

    quantization_section = config.setdefault("quantization", {})
    recommended = is_4bit_quantization_recommended()

    if "enabled" not in quantization_section:
        logger.info(
            "quantization.enabled absent du YAML — application de la "
            "recommandation matérielle : %s (GPU=%s).",
            recommended,
            get_gpu_name(),
        )
        quantization_section["enabled"] = recommended
        # Valeurs par défaut raisonnables si la section était totalement
        # absente — cohérentes avec ce que training_model_loader.py attend.
        quantization_section.setdefault("bnb_4bit_quant_type", "nf4")
        quantization_section.setdefault("bnb_4bit_use_double_quant", True)
    else:
        explicit = bool(quantization_section["enabled"])
        if explicit != recommended:
            logger.warning(
                "quantization.enabled=%s défini explicitement dans le "
                "YAML, alors que la recommandation matérielle pour ce "
                "GPU (%s) est %s. Choix explicite conservé — vérifier "
                "que c'est intentionnel (ex. marge de sécurité VRAM "
                "supplémentaire).",
                explicit,
                get_gpu_name(),
                recommended,
            )

    return config


def log_environment_info() -> KaggleEnvironment:
    """
    Log runtime information.
    """

    env = detect_environment()

    logger.info("========== TRAINING ENVIRONMENT ==========")
    logger.info("Kaggle Notebook: %s", env.is_kaggle)
    logger.info("Device        : %s", env.device)
    logger.info("CUDA Available: %s", env.cuda_available)
    logger.info("GPU           : %s", env.gpu_name)
    logger.info("BF16          : %s", env.bf16_supported)
    logger.info("FP16          : %s", env.fp16_supported)
    logger.info("Torch Version : %s", env.torch_version)
    logger.info("CUDA Version  : %s", env.cuda_version)
    logger.info("Training DType: %s", get_training_dtype())
    logger.info(
        "4bit Quantization Recommended: %s",
        is_4bit_quantization_recommended(),
    )
    logger.info("==========================================")

    return env


def detect_environment() -> KaggleEnvironment:
    """
    Build runtime environment descriptor.
    """

    return KaggleEnvironment(
        is_kaggle=is_running_in_kaggle(),
        cuda_available=torch.cuda.is_available(),
        device=get_device(),
        bf16_supported=is_bf16_supported(),
        fp16_supported=is_fp16_supported(),
        torch_version=torch.__version__,
        cuda_version=(
            torch.version.cuda
            if torch.cuda.is_available()
            else None
        ),
        gpu_name=get_gpu_name(),
    )


def get_training_arguments_precision() -> dict[str, Any]:
    """
    Return precision arguments compatible with
    Hugging Face TrainingArguments.

    The returned dictionary can safely be merged
    into the TrainingArguments keyword arguments.
    """

    cuda_available = torch.cuda.is_available()
    bf16 = is_bf16_supported()

    # NOTE : ce guard est en pratique redondant — is_bf16_supported()
    # s'appuie sur gpu_info.bf16_recommended (kaggle_gpu_detector.py),
    # dont la liste {L4, A100} exclut déjà T4/P100. bf16 ne peut donc
    # jamais être True ici pour un T4. Conservé comme filet de sécurité
    # explicite si la logique de bf16_recommended venait à changer.
    if bf16:
        gpu_name = get_gpu_name() or ""
        if "T4" in gpu_name or "Tesla" in gpu_name or "P100" in gpu_name:
            logger.warning(
                "BF16 détecté sur %s — GPU non supporté nativement. "
                "Forçage fp16.",
                gpu_name,
            )
            bf16 = False

    return {
        "bf16": bf16,
        "fp16": cuda_available and not bf16,
        "use_cpu": not cuda_available,
    }


def apply_precision_arguments(
    training_args: dict[str, Any],
) -> dict[str, Any]:
    """
    Inject runtime precision policy into a
    TrainingArguments configuration dictionary.
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
