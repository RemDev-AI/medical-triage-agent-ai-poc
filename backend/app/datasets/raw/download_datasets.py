# medical-triage-agent-ai-poc/backend/app/datasets/raw/download_datasets.py

"""
Téléchargement datasets RAW.
"""

from pathlib import Path

from backend.utils.hf_utils import (
    load_hf_dataset,
    export_jsonl,
    dataset_stats,
)

from raw.dataset_registry import (
    DATASET_REGISTRY,
)

RAW_DATA_DIR = Path("backend/app/datasets/raw/data")


def main():

    RAW_DATA_DIR.mkdir(parents=True, exist_ok=True)

    for dataset_name, config in DATASET_REGISTRY.items():

        print(f"\nDownloading: {dataset_name}")

        dataset = load_hf_dataset(
            dataset_name=config["hf_path"],
            subset=config["subset"],
        )

        output_path = (
            RAW_DATA_DIR /
            f"{dataset_name}.jsonl"
        )

        if hasattr(dataset, "keys"):
            train_dataset = dataset["train"]
        else:
            train_dataset = dataset

        export_jsonl(
            train_dataset,
            str(output_path),
        )

        stats = dataset_stats(train_dataset)

        print(stats)


if __name__ == "__main__":
    main()
