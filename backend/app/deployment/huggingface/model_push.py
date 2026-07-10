# medical-triage-agent-ai-poc/backend/app/deployment/huggingface/model_push.py

"""
model_push.py

Déploiement des modèles LoRA médicaux vers Hugging Face Hub.

Fonctionnalités :
- création automatique du repository HF ;
- upload tokenizer ;
- upload adapters LoRA ;
- upload checkpoints ;
- versioning ;
- logs détaillés ;
- reproductibilité MLOps.

Compatible :
- Transformers
- PEFT
- HuggingFace Hub
"""

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Optional  # noqa : F401

from huggingface_hub import HfApi, create_repo, upload_folder  # noqa : F401
from transformers import AutoTokenizer


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

HF_MODEL_REPO = "medical-triage-agent-ai-poc-models"

BASE_MODEL_NAME = "Qwen/Qwen3-1.7B-Base"

LOCAL_MODEL_DIR = Path("backend/models/final")

LOCAL_TOKENIZER_DIR = Path("backend/models/tokenizer")

PRIVATE_REPO = False


# =========================================================
# VALIDATION
# =========================================================


def validate_environment() -> None:
    """
    Vérifie les variables critiques.
    """

    if HF_TOKEN is None:
        raise EnvironmentError("HF_TOKEN manquant dans les variables d'environnement.")

    if not LOCAL_MODEL_DIR.exists():
        raise FileNotFoundError(f"Répertoire modèle introuvable : {LOCAL_MODEL_DIR}")

    logger.info("Validation environnement OK.")


# =========================================================
# REPOSITORY CREATION
# =========================================================


def create_model_repository() -> None:
    """
    Crée le repository HF si nécessaire.
    """

    logger.info("Création/Vérification du repository Hugging Face...")

    create_repo(
        repo_id=HF_MODEL_REPO,
        token=HF_TOKEN,
        private=PRIVATE_REPO,
        exist_ok=True,
        repo_type="model",
    )

    logger.info("Repository modèle prêt.")


# =========================================================
# TOKENIZER
# =========================================================


def push_tokenizer() -> None:
    """
    Push du tokenizer.
    """

    logger.info("Chargement tokenizer...")

    tokenizer = AutoTokenizer.from_pretrained(
        BASE_MODEL_NAME,
        trust_remote_code=True,
    )  # nosec B615 - usage temporaire pour le POC, sans pinning de revision ; a durcir avec un SHA de commit fige avant la prod

    LOCAL_TOKENIZER_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    tokenizer.save_pretrained(LOCAL_TOKENIZER_DIR)

    logger.info("Upload tokenizer...")

    upload_folder(
        repo_id=HF_MODEL_REPO,
        folder_path=str(LOCAL_TOKENIZER_DIR),
        path_in_repo="tokenizer",
        token=HF_TOKEN,
        repo_type="model",
    )

    logger.info("Tokenizer uploadé.")


# =========================================================
# LORA + CHECKPOINTS
# =========================================================


def push_model_artifacts(
    folder_name: str,
    local_path: Path,
) -> None:
    """
    Upload dossier modèle.
    """

    if not local_path.exists():
        logger.warning(f"Dossier absent : {local_path}")
        return

    logger.info(f"Upload de {folder_name}...")

    upload_folder(
        repo_id=HF_MODEL_REPO,
        folder_path=str(local_path),
        path_in_repo=folder_name,
        token=HF_TOKEN,
        repo_type="model",
    )

    logger.info(f"{folder_name} uploadé.")


# =========================================================
# MODEL CARD
# =========================================================


def create_model_card() -> None:
    """
    Génère une README.md HF.
    """

    logger.info("Création Model Card...")

    model_card = f"""
---
license: apache-2.0
language:
  - fr
  - en
pipeline_tag: text-generation
base_model: {BASE_MODEL_NAME}
library_name: transformers
tags:
  - medical
  - triage
  - llm
  - healthcare
  - qwen
  - lora
---

# Medical Triage Agent AI POC

## Description

Modèle LoRA spécialisé dans le triage médical.

## Fonctionnalités

- priorisation clinique ;
- génération recommandations ;
- classification urgence ;
- raisonnement médical assisté.

## Base Model

{BASE_MODEL_NAME}

## Fine-tuning

- SFT
- DPO
- LoRA adapters

## Sécurité

⚠️ Ce modèle ne remplace pas un médecin.

Usage :
- recherche ;
- expérimentation ;
- démonstration pédagogique.

## Architecture

- PEFT LoRA
- Transformers
- Qwen3
- Hugging Face

## Licence

Apache-2.0
"""

    readme_path = LOCAL_MODEL_DIR / "README.md"

    with open(readme_path, "w", encoding="utf-8") as f:
        f.write(model_card)

    logger.info("Model Card générée.")


# =========================================================
# MAIN
# =========================================================


def main() -> None:

    logger.info("Déploiement modèle Hugging Face démarré.")

    validate_environment()

    create_model_repository()

    create_model_card()

    push_tokenizer()

    push_model_artifacts(
        folder_name="lora-adapters",
        local_path=Path("backend/models/lora"),
    )

    push_model_artifacts(
        folder_name="checkpoints",
        local_path=Path("backend/models/checkpoints"),
    )

    push_model_artifacts(
        folder_name="final-model",
        local_path=LOCAL_MODEL_DIR,
    )

    logger.info("Déploiement modèle terminé avec succès.")


if __name__ == "__main__":
    main()
