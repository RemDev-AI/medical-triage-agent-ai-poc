# medical-triage-agent-ai-poc/backend/app/training/sft/dataset_builder.py

"""
Construction dataset SFT médical.
"""

from __future__ import annotations

import hashlib
import json
import random
from pathlib import Path

from sklearn.model_selection import train_test_split

INPUT_DIR = Path(
    "backend/app/datasets/raw/standardized"
)

OUTPUT_DIR = Path(
    "backend/app/datasets/processed/sft"
)

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

TARGET_SAMPLE_SIZE = 5000

random.seed(42)


def generate_id(text: str) -> str:

    return hashlib.md5(
        text.encode("utf-8")
    ).hexdigest()


def confidence_score(record: dict) -> float:

    base_score = 0.80

    if len(record["response"]) > 120:
        base_score += 0.05

    if "source" in record:
        base_score += 0.05

    return min(base_score, 0.99)


def clinical_tags(record: dict) -> list[str]:

    text = (
        record["instruction"] +
        " " +
        record["response"]
    ).lower()

    tags = []

    keywords = {
        "cardiology": ["coeur", "thoracique"],
        "respiratory": ["respiration", "toux"],
        "emergency": ["urgence", "douleur"],
        "neurology": ["migraine", "cerveau"],
    }

    for tag, words in keywords.items():

        if any(word in text for word in words):
            tags.append(tag)

    return tags


def deduplicate(records: list[dict]) -> list[dict]:

    seen = set()
    unique = []

    for record in records:

        signature = hashlib.md5(
            (
                record["instruction"] +
                record["response"]
            ).encode("utf-8")
        ).hexdigest()

        if signature not in seen:
            seen.add(signature)
            unique.append(record)

    return unique


def load_standardized_datasets():

    records = []

    for file_path in INPUT_DIR.glob("*.jsonl"):

        with open(file_path, "r", encoding="utf-8") as f:

            for line in f:

                item = json.loads(line)

                instruction = item.get(
                    "instruction",
                    ""
                ).strip()

                response = item.get(
                    "response",
                    ""
                ).strip()

                if not instruction or not response:
                    continue

                sft_record = {
                    "id": generate_id(
                        instruction + response
                    ),
                    "instruction": instruction,
                    "response": response,
                    "source": item.get("source"),
                    "language": item.get(
                        "language",
                        "fr",
                    ),
                    "confidence_score":
                        confidence_score(item),
                    "clinical_tags":
                        clinical_tags(item),
                    "metadata": item.get(
                        "metadata",
                        {},
                    ),
                }

                records.append(sft_record)

    return records


def save_jsonl(records, path):

    with open(path, "w", encoding="utf-8") as f:

        for record in records:
            f.write(
                json.dumps(
                    record,
                    ensure_ascii=False,
                ) + "\n"
            )


def build_splits(records):

    train, temp = train_test_split(
        records,
        test_size=0.20,
        random_state=42,
    )

    validation, test = train_test_split(
        temp,
        test_size=0.50,
        random_state=42,
    )

    clinical_eval = test[: min(100, len(test))]

    return (
        train,
        validation,
        test,
        clinical_eval,
    )


def main():

    print("\nLoading standardized datasets...")

    records = load_standardized_datasets()

    print(f"Loaded records: {len(records)}")

    records = deduplicate(records)

    print(f"After deduplication: {len(records)}")

    records = records[:TARGET_SAMPLE_SIZE]

    (
        train,
        validation,
        test,
        clinical_eval,
    ) = build_splits(records)

    save_jsonl(
        train,
        OUTPUT_DIR / "train.jsonl",
    )

    save_jsonl(
        validation,
        OUTPUT_DIR / "validation.jsonl",
    )

    save_jsonl(
        test,
        OUTPUT_DIR / "test.jsonl",
    )

    save_jsonl(
        clinical_eval,
        OUTPUT_DIR / "clinical_eval.jsonl",
    )

    print("\nSFT dataset build completed")


if __name__ == "__main__":
    main()
