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
from collections import Counter, defaultdict
from pathlib import Path

from sklearn.model_selection import train_test_split

from app.anonymization.presidio_anonymizer import anonymize_text

INPUT_DIR = Path("backend/app/datasets/raw/standardized")
OUTPUT_DIR = Path("backend/app/datasets/processed/sft")

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

TARGET_SAMPLE_SIZE = 5000
RANDOM_SEED = 42

random.seed(RANDOM_SEED)

# ==========================================================
# ADDITIONAL RGPD PATTERNS
# ==========================================================

EMAIL_PATTERN = re.compile(r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}")

PHONE_PATTERN = re.compile(r"(?:\+?\d[\d\s().-]{7,}\d)")

IP_PATTERN = re.compile(r"\b(?:\d{1,3}\.){3}\d{1,3}\b")

URL_PATTERN = re.compile(
    r"https?://[^\s]+|www\.[^\s]+",
    re.IGNORECASE,
)

SSN_PATTERN = re.compile(r"\b\d{3}-\d{2}-\d{4}\b")

MRN_PATTERN = re.compile(r"\b(?:MRN|mrn)[:\s#-]*[A-Z0-9]{5,20}\b")

PATIENT_ID_PATTERN = re.compile(
    r"\b(?:PATIENT|Patient|patient)[-_ ]?(?:ID|Id|id)?[:\s#-]*[A-Z0-9]{4,20}\b"
)

# Reliquats observés :
# Monsieur CAM.
# Monsieur RAT.
# Monsieur BOU...
# Dr. DUR.
PARTIAL_NAME_PATTERN = re.compile(
    r"\b(Monsieur|Madame|Mr\.|Mrs\.|Dr\.|Docteur)\s+[A-Z]{2,}\.(?:\s+[A-Z][a-z]+)?",  # noqa : E501
    re.IGNORECASE,
)

# Reliquats observés :
# LUC...Jean
# DUR...Pierre
TRUNCATED_NAME_PATTERN = re.compile(r"\b[A-Z]{3,}\.\.\.[A-Z][a-z]+\b")

CORRUPTED_MEDICAL_PATTERNS = [
    re.compile(
        r"\[PERSON\]\s+(?:pernicieux|viscéral|de)",
        re.IGNORECASE,
    ),
    re.compile(
        r"Réponse correcte\s*:\s*\[",
        re.IGNORECASE,
    ),
]


def contains_corrupted_medical_content(
    text: str,
) -> bool:

    if not text:
        return False

    return any(pattern.search(text) for pattern in CORRUPTED_MEDICAL_PATTERNS)


def generate_id(text: str) -> str:
    return hashlib.md5(  # nosec B324 - usage non cryptographique : simple identifiant déterministe
        text.encode("utf-8"),
        usedforsecurity=False,
    ).hexdigest()


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
    text = (record.get("instruction", "") + " " + record.get("response", "")).lower()

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
        if any(word in text for word in words):
            tags.append(tag)

    return tags


def deduplicate(records: list[dict]) -> list[dict]:
    seen = set()
    unique = []

    for record in records:
        signature = hashlib.md5(  # nosec B324 - usage non cryptographique : signature de déduplication, pas de sécurité requise
            (record["instruction"] + record["response"]).encode("utf-8"),
            usedforsecurity=False,
        ).hexdigest()

        if signature not in seen:
            seen.add(signature)
            unique.append(record)

    return unique


def contains_residual_pii(
    text: str,
) -> bool:

    if not text:
        return False

    patterns = (
        EMAIL_PATTERN,
        PHONE_PATTERN,
        IP_PATTERN,
        URL_PATTERN,
        SSN_PATTERN,
        MRN_PATTERN,
        PATIENT_ID_PATTERN,
        PARTIAL_NAME_PATTERN,
        TRUNCATED_NAME_PATTERN,
    )

    return any(pattern.search(text) for pattern in patterns)


def detect_residual_pii_types(
    text: str,
) -> list[str]:
    """
    Retourne les types de PII encore présents
    après anonymisation.

    Utilisé uniquement pour audit RGPD.
    """

    if not text:
        return []

    matches = []

    if EMAIL_PATTERN.search(text):
        matches.append("EMAIL")

    if PHONE_PATTERN.search(text):
        matches.append("PHONE")

    if IP_PATTERN.search(text):
        matches.append("IP_ADDRESS")

    if URL_PATTERN.search(text):
        matches.append("URL")

    if SSN_PATTERN.search(text):
        matches.append("US_SOCIAL_SECURITY")

    if MRN_PATTERN.search(text):
        matches.append("MEDICAL_RECORD_NUMBER")

    if PATIENT_ID_PATTERN.search(text):
        matches.append("PATIENT_ID")

    if PARTIAL_NAME_PATTERN.search(text):
        matches.append("PARTIAL_NAME")

    if TRUNCATED_NAME_PATTERN.search(text):
        matches.append("TRUNCATED_NAME")

    return matches


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


def sanitize_partial_names(
    text: str,
) -> str:
    """
    Nettoyage complémentaire après Presidio.

    Capture les reliquats observés :
    - Monsieur CAM.
    - Monsieur RAT.
    - Madame BOU...
    - Dr. DUR.
    - LUC...Jean
    """

    if not text:
        return text

    text = PARTIAL_NAME_PATTERN.sub(
        lambda m: f"{m.group(1)} [PERSON]",
        text,
    )

    text = TRUNCATED_NAME_PATTERN.sub(
        "[PERSON]",
        text,
    )

    return text


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

                metadata = dict(item.get("metadata", {}))

                language = item.get(
                    "language",
                    metadata.get(
                        "language",
                        None,
                    ),
                )

                instruction = item.get(
                    "instruction",
                    "",
                ).strip()

                response = item.get(
                    "response",
                    "",
                ).strip()

                if not instruction or not response:
                    continue

                records.append(
                    {
                        "id": generate_id(instruction + response),
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

    sampled = fr_records[:fr_target] + en_records[:en_target]

    random.shuffle(sampled)

    return sampled


def anonymize_records(
    records: list[dict],
) -> list[dict]:

    anonymized_records = []
    skipped_residual_pii = 0
    removal_stats = Counter()
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

        instruction = sanitize_partial_names(instruction)

        response = sanitize_partial_names(response)

        instruction_pii = detect_residual_pii_types(instruction)

        response_pii = detect_residual_pii_types(response)

        if instruction_pii or response_pii:

            skipped_residual_pii += 1

            for pii_type in instruction_pii + response_pii:
                removal_stats[pii_type] += 1

            print(f"[PII] {instruction_pii + response_pii}")

            continue

        if contains_corrupted_medical_content(
            instruction
        ) or contains_corrupted_medical_content(response):

            skipped_residual_pii += 1

            removal_stats["CORRUPTED_MEDICAL_CONTENT"] += 1

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
                f"[ANONYMIZATION] " f"{index:,}/{len(records):,} " f"| {elapsed:.1f}s"
            )

    print(f"\nResidual PII removed: " f"{skipped_residual_pii}")

    if removal_stats:

        print("\n=== RGPD AUDIT REPORT ===")

        total = sum(removal_stats.values())

        for pii_type, count in sorted(removal_stats.items()):

            percentage = (count / total) * 100

            print(f"{pii_type:<30} " f"{count:>6} " f"({percentage:5.2f}%)")

        print("=" * 40)

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
                )
                + "\n"
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

    print(f"After deduplication: " f"{len(records):,}")

    print(f"Deduplication time: " f"{time.time() - dedup_start:.2f}s")

    sampling_start = time.time()

    records = balanced_sampling(records)

    print(f"After sampling: " f"{len(records):,}")

    print(f"Sampling time: " f"{time.time() - sampling_start:.2f}s")

    anonymization_start = time.time()

    records = anonymize_records(records)

    print(f"Anonymization time: " f"{time.time() - anonymization_start:.2f}s")

    train, validation, test, clinical_eval = build_splits(records)

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

    total_time = time.time() - pipeline_start

    print(
        f"\nTotal pipeline time: " f"{total_time:.2f}s " f"({total_time / 60:.2f} min)"
    )

    print("\nSFT dataset build completed")


if __name__ == "__main__":
    main()
