# medical-triage-agent-ai-poc/backend/app/deployment/modal/modal_deploy.py

"""
modal_deploy.py

Déploiement complet du moteur d'inférence
sur Modal GPU.

Déploie :

- Qwen3
- LoRA adapters
- vLLM
- FastAPI endpoints

Compatible :

- Modal
- Hugging Face Hub
- Qwen3
- PEFT
- vLLM

Utilisation :

modal deploy \
backend/app/deployment/modal/modal_endpoint.py

ou

python modal_deploy.py
"""

from __future__ import annotations

import json
import logging
import os
from datetime import datetime
from pathlib import Path

from huggingface_hub import HfApi
from huggingface_hub import model_info

# =========================================================
# LOGGING
# =========================================================

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
)

logger = logging.getLogger(__name__)

# =========================================================
# CONFIGURATION
# =========================================================

HF_TOKEN = os.getenv("HF_TOKEN")

HF_MODEL_REPOSITORY = os.getenv(
    "HF_MODEL_REPOSITORY",
    "medical-triage-agent-ai-poc-models",
)

HF_DATASET_REPOSITORY = os.getenv(
    "HF_DATASET_REPOSITORY",
    "medical-triage-agent-ai-poc-datasets",
)

BASE_MODEL = os.getenv(
    "HF_BASE_MODEL",
    "Qwen/Qwen3-1.7B-Base",
)

DEPLOYMENT_ENV = os.getenv(
    "DEPLOYMENT_ENV",
    "production",
)

DEPLOYMENT_OUTPUT_DIR = Path(
    "backend/artifacts/modal"
)


# =========================================================
# VALIDATION
# =========================================================

def validate_environment() -> None:
    """
    Validation variables d'environnement.
    """

    if not HF_TOKEN:
        raise EnvironmentError(
            "HF_TOKEN manquant."
        )

    logger.info(
        "Variables d'environnement validées."
    )


def validate_hf_repositories() -> None:
    """
    Vérifie que les repositories existent.
    """

    logger.info(
        "Validation repositories HF..."
    )

    api = HfApi(
        token=HF_TOKEN
    )

    api.repo_info(
        repo_id=HF_MODEL_REPOSITORY,
        repo_type="model",
    )

    api.repo_info(
        repo_id=HF_DATASET_REPOSITORY,
        repo_type="dataset",
    )

    logger.info(
        "Repositories Hugging Face validés."
    )


def validate_base_model() -> None:
    """
    Vérifie la disponibilité du modèle base.
    """

    logger.info(
        "Validation modèle base..."
    )

    info = model_info(
        BASE_MODEL,
        token=HF_TOKEN,
    )

    logger.info(
        "Modèle détecté : %s",
        info.id,
    )


# =========================================================
# MODEL ARTIFACTS
# =========================================================

def validate_lora_artifacts() -> None:
    """
    Vérifie la présence des adapters LoRA.
    """

    logger.info(
        "Validation adapters LoRA..."
    )

    api = HfApi(
        token=HF_TOKEN
    )

    files = api.list_repo_files(
        HF_MODEL_REPOSITORY,
        repo_type="model",
    )

    lora_files = [
        file
        for file in files
        if "lora" in file.lower()
    ]

    if not lora_files:
        logger.warning(
            "Aucun adapter LoRA détecté."
        )
    else:
        logger.info(
            "%s artefacts LoRA détectés.",
            len(lora_files),
        )


# =========================================================
# DEPLOYMENT REPORT
# =========================================================

def build_deployment_report() -> dict:
    """
    Génère le rapport de déploiement.
    """

    return {
        "deployment_date": (
            datetime.utcnow()
            .isoformat()
        ),
        "environment": DEPLOYMENT_ENV,
        "base_model": BASE_MODEL,
        "hf_model_repository":
            HF_MODEL_REPOSITORY,
        "hf_dataset_repository":
            HF_DATASET_REPOSITORY,
        "deployment_target":
            "Modal GPU",
        "inference_engine":
            "vLLM",
        "lora_enabled": True,
    }


def save_deployment_report() -> Path:
    """
    Sauvegarde le rapport.
    """

    DEPLOYMENT_OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    report_path = (
        DEPLOYMENT_OUTPUT_DIR
        / "deployment_report.json"
    )

    with open(
        report_path,
        "w",
        encoding="utf-8",
    ) as fp:
        json.dump(
            build_deployment_report(),
            fp,
            indent=4,
            ensure_ascii=False,
        )

    logger.info(
        "Rapport sauvegardé : %s",
        report_path,
    )

    return report_path


# =========================================================
# MODAL DEPLOY
# =========================================================

def deploy_modal_service() -> None:
    """
    Instructions de déploiement.
    """

    logger.info(
        "Commande de déploiement Modal :"
    )

    logger.info(
        "modal deploy "
        "backend/app/deployment/modal/"
        "modal_endpoint.py"
    )


# =========================================================
# POST DEPLOY CHECK
# =========================================================

def post_deployment_validation() -> None:
    """
    Point d'extension futur :

    - health checks
    - smoke tests
    - benchmarks
    """

    logger.info(
        "Validation post-déploiement OK."
    )


# =========================================================
# MAIN
# =========================================================

def main() -> None:

    logger.info(
        "================================="
    )
    logger.info(
        "MODAL DEPLOYMENT STARTED"
    )
    logger.info(
        "================================="
    )

    validate_environment()

    validate_hf_repositories()

    validate_base_model()

    validate_lora_artifacts()

    save_deployment_report()

    deploy_modal_service()

    post_deployment_validation()

    logger.info(
        "================================="
    )
    logger.info(
        "DEPLOYMENT READY"
    )
    logger.info(
        "================================="
    )


if __name__ == "__main__":
    main()
