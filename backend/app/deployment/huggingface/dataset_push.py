# medical-triage-agent-ai-poc/backend/app/deployment/huggingface/dataset_push.py

"""
dataset_push.py

Déploiement datasets médicaux vers Hugging Face Hub.

Fonctionnalités :
- upload datasets ;
- génération Dataset Card ;
- métadonnées ;
- versioning ;
- conformité licences.

Compatible :
- datasets
- huggingface_hub
"""

from __future__ import annotations

import json
import logging
import os
from pathlib import Path

from datasets import DatasetDict, load_dataset  # noqa : F401
from huggingface_hub import create_repo, upload_folder

# =========================================================
# LOGGING
# =========================================================

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
)

logger = logging.getLogger(__name__)


# =========================================================
# CONFIG
# =========================================================

HF_TOKEN = os.getenv("HF_TOKEN")

HF_DATASET_REPO = "medical-triage-agent-ai-poc-datasets"

DATASET_DIR = Path("backend/app/datasets/processed")

PRIVATE_REPO = False


# =========================================================
# VALIDATION
# =========================================================


def validate_environment() -> None:

    if HF_TOKEN is None:
        raise EnvironmentError("HF_TOKEN manquant.")

    if not DATASET_DIR.exists():
        raise FileNotFoundError(f"Dataset introuvable : {DATASET_DIR}")

    logger.info("Validation datasets OK.")


# =========================================================
# REPO HF
# =========================================================


def create_dataset_repository() -> None:

    logger.info("Création/Vérification repository dataset...")

    create_repo(
        repo_id=HF_DATASET_REPO,
        token=HF_TOKEN,
        private=PRIVATE_REPO,
        exist_ok=True,
        repo_type="dataset",
    )

    logger.info("Repository dataset prêt.")


# =========================================================
# DATASET CARD
# =========================================================


def generate_dataset_card() -> None:

    logger.info("Création Dataset Card...")

    dataset_card = """
---
language:
  - fr
  - en
license: apache-2.0
task_categories:
  - text-generation
  - text-classification
pretty_name: Medical Triage Dataset
size_categories:
  - 1K<n<10K
---

# Medical Triage Dataset

## Description

Dataset spécialisé dans :
- triage médical ;
- priorisation clinique ;
- conversations patients ;
- recommandations médicales.

## Structure

Le dataset contient :
- instruction ;
- input ;
- output ;
- metadata ;
- confidence_score.

## Splits

- train
- validation
- test
- clinical_evaluation

## Sécurité

Toutes les données doivent être anonymisées.

## Sources

- MediQA
- MedQuAD
- FrenchMedMCQA
- UltraMedical

## Usage

Recherche IA médicale uniquement.

## Limitations

⚠️ Ne pas utiliser pour diagnostic clinique réel.
"""

    readme_path = DATASET_DIR / "README.md"

    with open(readme_path, "w", encoding="utf-8") as f:
        f.write(dataset_card)

    logger.info("Dataset Card générée.")


# =========================================================
# METADATA
# =========================================================


def generate_metadata() -> None:

    metadata = {
        "project": "Medical Triage Agent AI POC",
        "version": "1.0.0",
        "language": ["fr", "en"],
        "license": "apache-2.0",
        "pipeline": [
            "anonymization",
            "sft",
            "dpo",
            "triage",
        ],
    }

    metadata_path = DATASET_DIR / "metadata.json"

    with open(
        metadata_path,
        "w",
        encoding="utf-8",
    ) as f:
        json.dump(
            metadata,
            f,
            indent=4,
            ensure_ascii=False,
        )

    logger.info("Métadonnées générées.")


# =========================================================
# PUSH DATASET
# =========================================================


def push_dataset() -> None:

    logger.info("Upload datasets...")

    upload_folder(
        repo_id=HF_DATASET_REPO,
        folder_path=str(DATASET_DIR),
        path_in_repo="dataset",
        token=HF_TOKEN,
        repo_type="dataset",
    )

    logger.info("Datasets uploadés.")


# =========================================================
# MAIN
# =========================================================


def main() -> None:

    logger.info("Déploiement dataset Hugging Face démarré.")

    validate_environment()

    create_dataset_repository()

    generate_dataset_card()

    generate_metadata()

    push_dataset()

    logger.info("Déploiement dataset terminé.")


if __name__ == "__main__":
    main()
