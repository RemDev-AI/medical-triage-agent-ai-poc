# medical-triage-agent-ai-poc/backend/utils/hf_utils.py

"""
hf_utils.py

Utilitaires Hugging Face pour :
- téléchargement datasets ;
- export JSONL ;
- validation structure ;
- logging pipeline datasets.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Dict, Any

from datasets import Dataset, DatasetDict, load_dataset  # noqa : F401

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
)

logger = logging.getLogger(__name__)


def load_hf_dataset(
    dataset_name: str,
    subset: str | None = None,
    split: str | None = None,
):
    """
    Charge un dataset Hugging Face.

    Args:
        dataset_name: nom du dataset HF.
        subset: subset éventuel.
        split: split spécifique.

    Returns:
        Dataset ou DatasetDict.
    """

    logger.info(f"Loading dataset: {dataset_name}")

    if subset:
        return load_dataset(dataset_name, subset, split=split)

    return load_dataset(dataset_name, split=split)


def export_jsonl(
    dataset: Dataset,
    output_path: str,
):
    """
    Exporte un dataset HF au format JSONL UTF-8.
    """

    output_file = Path(output_path)
    output_file.parent.mkdir(parents=True, exist_ok=True)

    logger.info(f"Export JSONL -> {output_path}")

    with output_file.open("w", encoding="utf-8") as f:
        for row in dataset:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    logger.info("Export completed")


def validate_columns(
    dataset: Dataset,
    required_columns: list[str],
):
    """
    Vérifie la présence des colonnes obligatoires.
    """

    missing = [
        col for col in required_columns
        if col not in dataset.column_names
    ]

    if missing:
        raise ValueError(
            f"Missing required columns: {missing}"
        )

    logger.info("Dataset schema validated")


def dataset_stats(dataset: Dataset) -> Dict[str, Any]:
    """
    Génère statistiques simples.
    """

    return {
        "rows": len(dataset),
        "columns": dataset.column_names,
    }
