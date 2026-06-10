# medical-triage-agent-ai-poc/backend/app/training/sft/dataset_builder.py

"""
Construction dataset SFT médical bilingue.

Objectifs :
- Consolidation des datasets standardisés
- Déduplication
- Préservation des métadonnées
- Échantillonnage bilingue FR/EN
- Préparation SFT
"""

from __future__ import annotations

import hashlib
import json
import random
from collections import defaultdict
from pathlib import Path

from sklearn.model_selection import train_test_split

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


def load_standardized_datasets():

    records = []

    for file_path in sorted(
        INPUT_DIR.glob("*.jsonl")
    ):

        with open(
            file_path,
            "r",
            encoding="utf-8",
        ) as f:

            for line in f:

                item = json.loads(line)

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

                metadata = item.get(
                    "metadata",
                    {},
                )

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
                        item.get(
                            "language",
                            metadata.get(
                                "language",
                                "unknown",
                            ),
                        ),
                    "confidence_score":
                        confidence_score(
                            item
                        ),
                    "clinical_tags":
                        clinical_tags(
                            item
                        ),
                    "metadata":
                        metadata,
                }

                records.append(
                    sft_record
                )

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

    print(
        "\nLoading standardized datasets..."
    )

    records = (
        load_standardized_datasets()
    )

    print(
        f"Loaded records: "
        f"{len(records)}"
    )

    records = deduplicate(
        records
    )

    print(
        f"After deduplication: "
        f"{len(records)}"
    )

    records = balanced_sampling(
        records
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

    (
        train,
        validation,
        test,
        clinical_eval,
    ) = build_splits(
        records
    )

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

    print(
        "\nSFT dataset build completed"
    )


if __name__ == "__main__":
    main()
