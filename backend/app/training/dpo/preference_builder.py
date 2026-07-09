# medical-triage-agent-ai-poc/backend/app/training/dpo/preference_builder.py

"""
Construction dataset DPO médical bilingue.

Produit des paires :

- prompt
- chosen
- rejected

à partir du dataset SFT anonymisé.

Garanties :
- Validation PII résiduelle
- Compatibilité RGPD
- Monitoring complet
- Profilage détaillé
- Ré-anonymisation optionnelle
"""

from __future__ import annotations

import hashlib
import json
import random
import re
import time
from pathlib import Path

from sklearn.model_selection import train_test_split

from backend.app.anonymization.presidio_anonymizer import (
    anonymize_text,
)

INPUT_DIR = Path("backend/app/datasets/processed/sft")

OUTPUT_DIR = Path("backend/app/datasets/processed/dpo")

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True,
)

RANDOM_SEED = 42

random.seed(RANDOM_SEED)


# ==========================================================
# DPO CONFIG
# ==========================================================

REANONYMIZE_DPO = False

PROGRESS_INTERVAL = 500

SLOW_PAIR_THRESHOLD = 0.5


# ==========================================================
# COMPILED REGEX
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


# ==========================================================
# PII VALIDATION
# ==========================================================


def contains_residual_pii(
    text: str,
) -> bool:
    """
    Validation défensive post-anonymisation.

    Objectif :
    détecter une éventuelle fuite résiduelle
    qui aurait échappé à Presidio.

    Cette validation ne remplace pas
    Presidio/spaCy.
    """

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
    )

    return any(pattern.search(text) for pattern in patterns)


# ==========================================================
# REJECTED GENERATION
# ==========================================================


def generate_rejected_response(
    response: str,
    language: str,
) -> str:
    """
    Génère une réponse volontairement
    moins informative.
    """

    degraded = response

    replacements = {
        "fr": {
            "urgence": "attendre",
            "urgent": "non urgent",
            "médecin": "internet",
            "consulter": "rechercher en ligne",
            "consultation": "automédication",
            "traitement": "solution",
            "diagnostic": "supposition",
        },
        "en": {
            "emergency": "minor issue",
            "urgent": "optional",
            "doctor": "internet",
            "physician": "internet",
            "consult": "search online",
            "treatment": "general advice",
            "diagnosis": "guess",
            "medical attention": "wait and see",
        },
    }.get(language, {})

    for source, target in replacements.items():

        degraded = degraded.replace(
            source,
            target,
        )

        degraded = degraded.replace(
            source.capitalize(),
            target.capitalize(),
        )

    if degraded == response:

        sentences = response.split(". ")

        if len(sentences) > 1:

            degraded = ". ".join(
                sentences[
                    : max(
                        1,
                        len(sentences) // 2,
                    )
                ]
            )

        else:

            degraded = (
                "Informations limitées."
                if language == "fr"
                else "Limited information available."
            )

    return degraded.strip()


# ==========================================================
# SCORING
# ==========================================================


def clinical_quality_score(
    response: str,
) -> float:

    score = 0.80

    positive_terms = [
        "doctor",
        "physician",
        "treatment",
        "diagnosis",
        "medical",
        "consult",
        "médecin",
        "traitement",
        "diagnostic",
        "consulter",
        "urgence",
        "emergency",
    ]

    text = response.lower()

    for term in positive_terms:

        if term in text:
            score += 0.02

    return min(score, 1.0)


def safety_score(
    response: str,
) -> float:

    score = 0.90

    dangerous_terms = [
        "ignore",
        "search online",
        "wait and see",
        "guess",
        "internet",
        "ignorer",
        "automédication",
        "attendre",
    ]

    text = response.lower()

    for term in dangerous_terms:

        if term in text:
            score -= 0.10

    return max(score, 0.0)


# ==========================================================
# OPTIONAL RE-ANONYMIZATION
# ==========================================================


def anonymize_preference_fields(
    prompt: str,
    chosen: str,
    rejected: str,
    language: str,
) -> tuple[str, str, str]:

    if not REANONYMIZE_DPO:
        return (
            prompt,
            chosen,
            rejected,
        )

    return (
        anonymize_text(
            prompt,
            language=language,
        ),
        anonymize_text(
            chosen,
            language=language,
        ),
        anonymize_text(
            rejected,
            language=language,
        ),
    )


# ==========================================================
# BUILD
# ==========================================================


def build_preferences():

    sft_train = INPUT_DIR / "train.jsonl"

    preferences = []

    skipped_residual_pii = 0
    processed = 0
    slow_pairs = 0

    rejected_time = 0.0
    anonymization_time = 0.0
    pii_time = 0.0

    build_start = time.time()

    with open(
        sft_train,
        "r",
        encoding="utf-8",
    ) as f:

        for line in f:

            pair_start = time.time()

            processed += 1

            item = json.loads(line)

            language = item.get(
                "language",
                "unknown",
            )

            prompt = item["instruction"].strip()

            chosen = item["response"].strip()

            # --------------------------------------
            # rejected
            # --------------------------------------

            t0 = time.time()

            rejected = generate_rejected_response(
                chosen,
                language,
            )

            rejected_time += time.time() - t0

            if rejected == chosen:
                continue

            # --------------------------------------
            # optional anonymization
            # --------------------------------------

            t0 = time.time()

            (
                prompt,
                chosen,
                rejected,
            ) = anonymize_preference_fields(
                prompt,
                chosen,
                rejected,
                language,
            )

            anonymization_time += time.time() - t0

            # --------------------------------------
            # pii validation
            # --------------------------------------

            t0 = time.time()

            has_pii = (
                contains_residual_pii(prompt)
                or contains_residual_pii(chosen)
                or contains_residual_pii(rejected)
            )

            pii_time += time.time() - t0

            if has_pii:

                skipped_residual_pii += 1

                continue

            metadata = dict(
                item.get(
                    "metadata",
                    {},
                )
            )

            metadata["anonymized"] = True

            preference_record = {
                "id": hashlib.md5((prompt + chosen).encode("utf-8")).hexdigest(),
                "prompt": prompt,
                "chosen": chosen,
                "rejected": rejected,
                "language": language,
                "clinical_quality_score": clinical_quality_score(chosen),
                "safety_score": safety_score(rejected),
                "source": item.get(
                    "source",
                    "unknown",
                ),
                "metadata": metadata,
            }

            preferences.append(preference_record)

            elapsed_pair = time.time() - pair_start

            if elapsed_pair > SLOW_PAIR_THRESHOLD:
                slow_pairs += 1

                print(
                    f"[SLOW] Pair "
                    f"{processed:,} "
                    f"processed in "
                    f"{elapsed_pair:.2f}s"
                )

            if processed % PROGRESS_INTERVAL == 0:

                elapsed = time.time() - build_start

                print(
                    f"[PROGRESS] "
                    f"{processed:,} pairs | "
                    f"{elapsed:.1f}s | "
                    f"{elapsed/processed:.4f}s/pair"
                )

    total_time = time.time() - build_start

    print("\n" + "=" * 60)
    print("DPO PIPELINE SUMMARY")
    print("=" * 60)
    print(f"Pairs processed: {processed:,}")
    print(f"Preferences generated: " f"{len(preferences):,}")
    print(f"Residual PII removed: " f"{skipped_residual_pii:,}")
    print(f"Slow pairs (>0.5s): " f"{slow_pairs:,}")
    print(f"REANONYMIZE_DPO: " f"{REANONYMIZE_DPO}")
    print(f"generate_rejected_response(): " f"{rejected_time:.2f}s")
    print(f"anonymize_preference_fields(): " f"{anonymization_time:.2f}s")
    print(f"contains_residual_pii(): " f"{pii_time:.2f}s")
    print(f"Average time per pair: " f"{total_time/max(processed, 1):.4f}s")
    print(f"Total build time: " f"{total_time:.2f}s")
    print("=" * 60)

    return preferences


# ==========================================================
# SAVE
# ==========================================================


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


# ==========================================================
# SPLIT
# ==========================================================


def split_preferences(
    records,
):

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


# ==========================================================
# MAIN
# ==========================================================


def main():

    pipeline_start = time.time()

    print("\nBuilding DPO preferences...")

    preferences = build_preferences()

    train, validation, test, clinical_eval = split_preferences(preferences)

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

    total_pipeline = time.time() - pipeline_start

    print(f"\nTotal pipeline time: " f"{total_pipeline:.2f}s")

    print("DPO dataset build completed")


if __name__ == "__main__":
    main()
