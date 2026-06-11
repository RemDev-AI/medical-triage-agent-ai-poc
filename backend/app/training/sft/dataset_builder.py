# medical-triage-agent-ai-poc/backend/app/training/sft/dataset_builder.py

"""
Construction dataset SFT médical bilingue.

Pipeline optimisé :
- Chargement des datasets standardisés
- Déduplication
- Échantillonnage bilingue FR/EN
- Anonymisation uniquement sur l'échantillon retenu
- Validation PII résiduelle
- Génération des splits SFT
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

from backend.app.anonymization.presidio_anonymizer import anonymize_text

INPUT_DIR = Path("backend/app/datasets/raw/standardized")
OUTPUT_DIR = Path("backend/app/datasets/processed/sft")

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

TARGET_SAMPLE_SIZE = 5000
RANDOM_SEED = 42

random.seed(RANDOM_SEED)


def generate_id(text: str) -> str:
    return hashlib.md5(text.encode("utf-8")).hexdigest()


def confidence_score(record: dict) -> float:
    score = 0.80

    response = record.get("response", "")

    if len(response) > 120:
        score += 0.05

    if record.get("source"):
        score += 0.05

    if record.get("language") in {"fr", "en"}:
        score += 0.03

    return min(score, 0.99)


def clinical_tags(record: dict) -> list[str]:
    text = (
        record.get("instruction", "") + " " +
        record.get("response", "")
    ).lower()

    keywords = {
        "cardiology": [
            "coeur", "heart", "cardiac",
            "thoracique", "chest pain",
        ],
        "respiratory": [
            "respiration", "breathing",
            "toux", "cough", "lung",
        ],
        "emergency": [
            "urgence", "emergency",
            "critical", "douleur sévère",
            "severe pain",
        ],
        "neurology": [
            "migraine", "stroke",
            "brain", "cerveau",
            "neurolog",
        ],
    }

    tags = []

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


def contains_residual_pii(text: str) -> bool:
    if not text:
        return False

    email_pattern = re.compile(
        r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}"
    )

    phone_pattern = re.compile(
        r"(?:\+?\d[\d\s().-]{7,}\d)"
    )

    return bool(
        email_pattern.search(text)
        or phone_pattern.search(text)
    )


def anonymize_record(
    instruction: str,
    response: str,
    language: str | None,
) -> tuple[str, str]:

    return (
        anonymize_text(
            instruction,
            language=language,
        ),
        anonymize_text(
            response,
            language=language,
        ),
    )


def load_standardized_datasets() -> list[dict]:
    records = []
    total_records = 0
    start_time = time.time()

    for file_path in sorted(INPUT_DIR.glob("*.jsonl")):

        print(f"\nProcessing file: {file_path.name}")

        with open(
            file_path,
            "r",
            encoding="utf-8",
        ) as f:

            for line in f:
                total_records += 1

                try:
                    item = json.loads(line)
                except Exception:
                    continue

                metadata = dict(
                    item.get("metadata", {})
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

                if not instruction or not response:
                    continue

                records.append(
                    {
                        "id": generate_id(
                            instruction + response
                        ),
                        "instruction": instruction,
                        "response": response,
                        "source": item.get("source"),
                        "language": language,
                        "confidence_score": confidence_score(item),
                        "clinical_tags": clinical_tags(
                            {
                                "instruction": instruction,
                                "response": response,
                            }
                        ),
                        "metadata": metadata,
                    }
                )

                if total_records % 10000 == 0:
                    elapsed = time.time() - start_time

                    print(
                        f"[PROGRESS] "
                        f"{total_records:,} records loaded | "
                        f"{elapsed:.1f}s"
                    )

    print(f"\nLoaded {len(records):,} valid records")

    return records


def balanced_sampling(
    records: list[dict],
) -> list[dict]:

    by_language = defaultdict(list)

    for record in records:
        by_language[
            record.get(
                "language",
                "unknown",
            )
        ].append(record)

    for lang_records in by_language.values():
        random.shuffle(lang_records)

    fr_records = by_language.get("fr", [])
    en_records = by_language.get("en", [])

    if not fr_records or not en_records:
        random.shuffle(records)
        return records[:TARGET_SAMPLE_SIZE]

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


def anonymize_records(
    records: list[dict],
) -> list[dict]:

    anonymized_records = []
    skipped_residual_pii = 0
    start_time = time.time()

    for index, record in enumerate(
        records,
        start=1,
    ):

        instruction, response = anonymize_record(
            instruction=record["instruction"],
            response=record["response"],
            language=record.get("language"),
        )

        if (
            contains_residual_pii(instruction)
            or contains_residual_pii(response)
        ):
            skipped_residual_pii += 1
            continue

        updated = dict(record)

        updated["instruction"] = instruction
        updated["response"] = response

        metadata = dict(
            updated.get(
                "metadata",
                {},
            )
        )

        metadata["anonymized"] = True
        updated["metadata"] = metadata

        anonymized_records.append(updated)

        if index % 1000 == 0:
            elapsed = time.time() - start_time

            print(
                f"[ANONYMIZATION] "
                f"{index:,}/{len(records):,} "
                f"| {elapsed:.1f}s"
            )

    print(
        f"Residual PII removed: "
        f"{skipped_residual_pii}"
    )

    return anonymized_records


def save_jsonl(records, path):
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
                ) + "\n"
            )


def build_splits(records):

    train, temp = train_test_split(
        records,
        test_size=0.20,
        random_state=RANDOM_SEED,
        shuffle=True,
    )

    validation, test = train_test_split(
        temp,
        test_size=0.50,
        random_state=RANDOM_SEED,
        shuffle=True,
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

    print("\nLoading standardized datasets...")

    records = load_standardized_datasets()

    print(f"\nLoaded records: {len(records):,}")

    dedup_start = time.time()

    records = deduplicate(records)

    print(
        f"After deduplication: "
        f"{len(records):,}"
    )

    print(
        f"Deduplication time: "
        f"{time.time() - dedup_start:.2f}s"
    )

    sampling_start = time.time()

    records = balanced_sampling(records)

    print(
        f"After sampling: "
        f"{len(records):,}"
    )

    print(
        f"Sampling time: "
        f"{time.time() - sampling_start:.2f}s"
    )

    anonymization_start = time.time()

    records = anonymize_records(records)

    print(
        f"Anonymization time: "
        f"{time.time() - anonymization_start:.2f}s"
    )

    train, validation, test, clinical_eval = (
        build_splits(records)
    )

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

    total_time = (
        time.time() - pipeline_start
    )

    print(
        f"\nTotal pipeline time: "
        f"{total_time:.2f}s "
        f"({total_time / 60:.2f} min)"
    )

    print(
        "\nSFT dataset build completed"
    )


if __name__ == "__main__":
    main()
