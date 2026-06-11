# medical-triage-agent-ai-poc/backend/app/training/sft/dataset_builder.py

"""
Construction dataset SFT médical bilingue.

Objectifs :
- Consolidation des datasets standardisés
- Anonymisation PII systématique
- Déduplication
- Préservation des métadonnées
- Échantillonnage bilingue FR/EN
- Préparation SFT
"""

from __future__ import annotations

import hashlib
import json
import random
import re
import time
from collections import defaultdict
from pathlib import Path

from sklearn.model_selection import train_test_split

from backend.app.anonymization.presidio_anonymizer import (
    anonymize_text,
)

INPUT_DIR = Path(
    "backend/app/datasets/raw/standardized"
)

OUTPUT_DIR = Path(
    "backend/app/datasets/processed/sft"
)

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True,
)

TARGET_SAMPLE_SIZE = 5000

RANDOM_SEED = 42

random.seed(RANDOM_SEED)


def generate_id(text: str) -> str:
    return hashlib.md5(
        text.encode("utf-8")
    ).hexdigest()


def confidence_score(record: dict) -> float:
    score = 0.80

    response = record.get(
        "response",
        "",
    )

    if len(response) > 120:
        score += 0.05

    if record.get("source"):
        score += 0.05

    if record.get("language") in {
        "fr",
        "en",
    }:
        score += 0.03

    return min(score, 0.99)


def clinical_tags(record: dict) -> list[str]:
    text = (
        record.get("instruction", "")
        + " "
        + record.get("response", "")
    ).lower()

    keywords = {
        "cardiology": [
            "coeur",
            "heart",
            "cardiac",
            "thoracique",
            "chest pain",
        ],
        "respiratory": [
            "respiration",
            "breathing",
            "toux",
            "cough",
            "lung",
        ],
        "emergency": [
            "urgence",
            "emergency",
            "critical",
            "douleur sévère",
            "severe pain",
        ],
        "neurology": [
            "migraine",
            "stroke",
            "brain",
            "cerveau",
            "neurolog",
        ],
    }

    tags = []

    for tag, words in keywords.items():

        if any(
            word in text
            for word in words
        ):
            tags.append(tag)

    return tags


def deduplicate(
    records: list[dict],
) -> list[dict]:

    seen = set()

    unique = []

    for record in records:

        signature = hashlib.md5(
            (
                record["instruction"]
                + record["response"]
            ).encode("utf-8")
        ).hexdigest()

        if signature not in seen:

            seen.add(signature)

            unique.append(record)

    return unique


def contains_residual_pii(
    text: str,
) -> bool:
    """
    Validation légère post-anonymisation.

    Détecte des patterns résiduels
    qui ne devraient plus apparaître.
    """

    if not text:
        return False

    email_pattern = re.compile(
        r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}"
    )

    phone_pattern = re.compile(
        r"(?:\+?\d[\d\s().-]{7,}\d)"
    )

    if email_pattern.search(text):
        return True

    if phone_pattern.search(text):
        return True

    return False


def anonymize_record(
    instruction: str,
    response: str,
    language: str | None,
) -> tuple[str, str]:

    anonymized_instruction = anonymize_text(
        instruction,
        language=language,
    )

    anonymized_response = anonymize_text(
        response,
        language=language,
    )

    return (
        anonymized_instruction,
        anonymized_response,
    )


def load_standardized_datasets():

    records = []

    skipped_residual_pii = 0

    total_records = 0

    total_anonymization_time = 0.0

    slow_anonymizations = 0

    global_start = time.time()

    for file_path in sorted(
        INPUT_DIR.glob("*.jsonl")
    ):

        print(
            f"\nProcessing file: {file_path.name}"
        )

        with open(
            file_path,
            "r",
            encoding="utf-8",
        ) as f:

            for line in f:

                total_records += 1

                try:
                    item = json.loads(line)

                except Exception as e:
                    print(
                        f"Invalid JSON at record "
                        f"{total_records}: {e}"
                    )
                    continue

                metadata = dict(
                    item.get(
                        "metadata",
                        {},
                    )
                )

                language = item.get(
                    "language",
                    metadata.get(
                        "language",
                        None,
                    ),
                )

                instruction = (
                    item.get(
                        "instruction",
                        "",
                    ).strip()
                )

                response = (
                    item.get(
                        "response",
                        "",
                    ).strip()
                )

                if not instruction:
                    continue

                if not response:
                    continue

                anonymization_start = (
                    time.time()
                )

                instruction, response = (
                    anonymize_record(
                        instruction=instruction,
                        response=response,
                        language=language,
                    )
                )

                anonymization_elapsed = (
                    time.time()
                    - anonymization_start
                )

                total_anonymization_time += (
                    anonymization_elapsed
                )

                if anonymization_elapsed > 0.5:

                    slow_anonymizations += 1

                    print(
                        f"[SLOW] Record "
                        f"{total_records} "
                        f"anonymized in "
                        f"{anonymization_elapsed:.2f}s"
                    )

                if (
                    contains_residual_pii(
                        instruction
                    )
                    or contains_residual_pii(
                        response
                    )
                ):
                    skipped_residual_pii += 1
                    continue

                metadata["anonymized"] = True

                sft_record = {
                    "id": generate_id(
                        instruction + response
                    ),
                    "instruction":
                        instruction,
                    "response":
                        response,
                    "source":
                        item.get(
                            "source"
                        ),
                    "language":
                        language,
                    "confidence_score":
                        confidence_score(
                            item
                        ),
                    "clinical_tags":
                        clinical_tags(
                            {
                                "instruction":
                                    instruction,
                                "response":
                                    response,
                            }
                        ),
                    "metadata":
                        metadata,
                }

                records.append(
                    sft_record
                )

                if total_records % 1000 == 0:

                    elapsed = (
                        time.time()
                        - global_start
                    )

                    avg_per_record = (
                        elapsed
                        / total_records
                    )

                    print(
                        f"[PROGRESS] "
                        f"{total_records:,} "
                        f"records processed | "
                        f"{elapsed:.1f}s elapsed | "
                        f"{avg_per_record:.4f}s/record"
                    )

    total_elapsed = (
        time.time()
        - global_start
    )

    avg_anonymization = (
        total_anonymization_time
        / total_records
        if total_records
        else 0
    )

    print("\n" + "=" * 60)
    print("DATASET LOADING SUMMARY")
    print("=" * 60)

    print(
        f"Total records read: "
        f"{total_records:,}"
    )

    print(
        f"Valid records kept: "
        f"{len(records):,}"
    )

    print(
        f"Residual PII removed: "
        f"{skipped_residual_pii:,}"
    )

    print(
        f"Slow anonymizations (>0.5s): "
        f"{slow_anonymizations:,}"
    )

    print(
        f"Total anonymization time: "
        f"{total_anonymization_time:.2f}s"
    )

    print(
        f"Average anonymization time: "
        f"{avg_anonymization:.4f}s"
    )

    print(
        f"Total loading time: "
        f"{total_elapsed:.2f}s"
    )

    print("=" * 60)

    return records


def balanced_sampling(
    records: list[dict],
) -> list[dict]:
    """
    Garantit la présence FR + EN.
    """

    by_language = defaultdict(list)

    for record in records:

        language = record.get(
            "language",
            "unknown",
        )

        by_language[
            language
        ].append(record)

    for lang_records in (
        by_language.values()
    ):
        random.shuffle(
            lang_records
        )

    fr_records = by_language.get(
        "fr",
        [],
    )

    en_records = by_language.get(
        "en",
        [],
    )

    if not fr_records or not en_records:

        random.shuffle(records)

        return records[
            :TARGET_SAMPLE_SIZE
        ]

    fr_target = min(
        len(fr_records),
        TARGET_SAMPLE_SIZE // 2,
    )

    en_target = min(
        len(en_records),
        TARGET_SAMPLE_SIZE - fr_target,
    )

    sampled = (
        fr_records[:fr_target]
        + en_records[:en_target]
    )

    random.shuffle(sampled)

    return sampled


def save_jsonl(
    records,
    path,
):

    with open(
        path,
        "w",
        encoding="utf-8",
    ) as f:

        for record in records:

            f.write(
                json.dumps(
                    record,
                    ensure_ascii=False,
                )
                + "\n"
            )


def build_splits(
    records,
):

    train, temp = (
        train_test_split(
            records,
            test_size=0.20,
            random_state=RANDOM_SEED,
            shuffle=True,
        )
    )

    validation, test = (
        train_test_split(
            temp,
            test_size=0.50,
            random_state=RANDOM_SEED,
            shuffle=True,
        )
    )

    clinical_eval = test[
        : min(
            100,
            len(test),
        )
    ]

    return (
        train,
        validation,
        test,
        clinical_eval,
    )


def main():

    pipeline_start = time.time()

    print(
        "\nLoading standardized datasets..."
    )

    records = (
        load_standardized_datasets()
    )

    print(
        f"\nLoaded records: "
        f"{len(records):,}"
    )

    dedup_start = time.time()

    records = deduplicate(
        records
    )

    dedup_time = (
        time.time()
        - dedup_start
    )

    print(
        f"After deduplication: "
        f"{len(records):,}"
    )

    print(
        f"Deduplication time: "
        f"{dedup_time:.2f}s"
    )

    sampling_start = time.time()

    records = balanced_sampling(
        records
    )

    sampling_time = (
        time.time()
        - sampling_start
    )

    print(
        f"Sampling time: "
        f"{sampling_time:.2f}s"
    )

    language_stats = defaultdict(
        int
    )

    for record in records:

        language_stats[
            record.get(
                "language",
                "unknown",
            )
        ] += 1

    print(
        "\nLanguage distribution:"
    )

    for lang, count in sorted(
        language_stats.items()
    ):
        print(
            f"  {lang}: {count}"
        )

    split_start = time.time()

    (
        train,
        validation,
        test,
        clinical_eval,
    ) = build_splits(
        records
    )

    split_time = (
        time.time()
        - split_start
    )

    print(
        f"\nSplit generation time: "
        f"{split_time:.2f}s"
    )

    save_start = time.time()

    save_jsonl(
        train,
        OUTPUT_DIR
        / "train.jsonl",
    )

    save_jsonl(
        validation,
        OUTPUT_DIR
        / "validation.jsonl",
    )

    save_jsonl(
        test,
        OUTPUT_DIR
        / "test.jsonl",
    )

    save_jsonl(
        clinical_eval,
        OUTPUT_DIR
        / "clinical_eval.jsonl",
    )

    save_time = (
        time.time()
        - save_start
    )

    print(
        f"Save time: "
        f"{save_time:.2f}s"
    )

    total_pipeline_time = (
        time.time()
        - pipeline_start
    )

    print("\n" + "=" * 60)
    print("PIPELINE SUMMARY")
    print("=" * 60)

    print(
        f"Total pipeline time: "
        f"{total_pipeline_time:.2f}s "
        f"({total_pipeline_time / 60:.2f} min)"
    )

    print("=" * 60)

    print(
        "\nSFT dataset build completed"
    )


if __name__ == "__main__":
    main()
