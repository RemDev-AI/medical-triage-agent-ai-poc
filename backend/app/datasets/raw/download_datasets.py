# medical-triage-agent-ai-poc/backend/app/datasets/raw/download_datasets.py

"""
Téléchargement datasets RAW.

Responsabilités :
- téléchargement depuis Hugging Face ;
- sélection automatique du split principal ;
- export JSONL ;
- génération de statistiques ;
- tolérance aux erreurs dataset par dataset.
"""

from pathlib import Path
import logging

from backend.utils.hf_utils import (
    load_hf_dataset,
    export_jsonl,
    dataset_stats,
)

from backend.app.datasets.raw.dataset_registry import (
    DATASET_REGISTRY,
)

logger = logging.getLogger(__name__)

RAW_DATA_DIR = Path(
    "backend/app/datasets/raw/data"
)


def get_primary_split(dataset):
    """
    Sélectionne automatiquement le split principal.

    Priorité :
    train > validation > test

    Args:
        dataset: Dataset ou DatasetDict

    Returns:
        Dataset
    """

    if not hasattr(dataset, "keys"):
        return dataset

    for split_name in (
        "train",
        "validation",
        "test",
    ):
        if split_name in dataset:
            logger.info(
                "Using split '%s'",
                split_name,
            )
            return dataset[split_name]

    raise ValueError(
        "No supported split found. "
        f"Available splits: {list(dataset.keys())}"
    )


def main():
    """
    Télécharge tous les datasets du registry.
    """

    RAW_DATA_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    success_count = 0
    failure_count = 0

    for dataset_name, config in DATASET_REGISTRY.items():

        print(f"\nDownloading: {dataset_name}")

        try:

            dataset = load_hf_dataset(
                dataset_name=config["hf_path"],
                subset=config.get("subset"),
            )

            selected_dataset = get_primary_split(
                dataset
            )

            output_path = (
                RAW_DATA_DIR /
                f"{dataset_name}.jsonl"
            )

            export_jsonl(
                selected_dataset,
                str(output_path),
            )

            stats = dataset_stats(
                selected_dataset
            )

            print(
                f"SUCCESS: {dataset_name}"
            )
            print(stats)

            success_count += 1

        except Exception as exc:

            failure_count += 1

            logger.exception(
                "Failed to process dataset '%s'",
                dataset_name,
            )

            print(
                f"ERROR: {dataset_name} -> {exc}"
            )

    print("\n====================")
    print("DOWNLOAD SUMMARY")
    print("====================")
    print(f"Success : {success_count}")
    print(f"Failed  : {failure_count}")


if __name__ == "__main__":
    main()
