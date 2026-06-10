# medical-triage-agent-ai-poc/backend/app/training/dpo/preference_builder.py

"""
Construction dataset DPO médical bilingue.

Produit des paires :

prompt
chosen
rejected

à partir du dataset SFT anonymisé.

Garanties :
- Ré-anonymisation défensive
- Validation PII résiduelle
- Conservation des métadonnées
- Compatibilité RGPD
"""

from __future__ import annotations

import hashlib
import json
import random
import re
from pathlib import Path

from sklearn.model_selection import train_test_split

from backend.app.anonymization.presidio_anonymizer import (
    anonymize_text,
)

INPUT_DIR = Path(
    "backend/app/datasets/processed/sft"
)

OUTPUT_DIR = Path(
    "backend/app/datasets/processed/dpo"
)

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True,
)

RANDOM_SEED = 42

random.seed(RANDOM_SEED)


def contains_residual_pii(
    text: str,
) -> bool:
    """
    Validation légère post-anonymisation.
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


def generate_rejected_response(
    response: str,
    language: str,
) -> str:
    """
    Génère une réponse volontairement
    moins informative.
    """

    degraded = response

    if language == "fr":

        replacements = {
            "urgence": "attendre",
            "urgent": "non urgent",
            "médecin": "internet",
            "consulter": "rechercher en ligne",
            "consultation": "automédication",
            "traitement": "solution",
            "diagnostic": "supposition",
        }

    else:

        replacements = {
            "emergency": "minor issue",
            "urgent": "optional",
            "doctor": "internet",
            "physician": "internet",
            "consult": "search online",
            "treatment": "general advice",
            "diagnosis": "guess",
            "medical attention": "wait and see",
        }

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
                "Limited information available."
                if language == "en"
                else "Informations limitées."
            )

    return degraded.strip()


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


def anonymize_preference_fields(
    prompt: str,
    chosen: str,
    rejected: str,
    language: str,
) -> tuple[str, str, str]:

    prompt = anonymize_text(
        prompt,
        language=language,
    )

    chosen = anonymize_text(
        chosen,
        language=language,
    )

    rejected = anonymize_text(
        rejected,
        language=language,
    )

    return (
        prompt,
        chosen,
        rejected,
    )


def build_preferences():

    sft_train = (
        INPUT_DIR / "train.jsonl"
    )

    preferences = []

    skipped_residual_pii = 0

    with open(
        sft_train,
        "r",
        encoding="utf-8",
    ) as f:

        for line in f:

            item = json.loads(line)

            language = item.get(
                "language",
                "unknown",
            )

            prompt = (
                item["instruction"]
                .strip()
            )

            chosen = (
                item["response"]
                .strip()
            )

            rejected = (
                generate_rejected_response(
                    chosen,
                    language,
                )
            )

            if rejected == chosen:
                continue

            (
                prompt,
                chosen,
                rejected,
            ) = anonymize_preference_fields(
                prompt=prompt,
                chosen=chosen,
                rejected=rejected,
                language=language,
            )

            if (
                contains_residual_pii(prompt)
                or contains_residual_pii(chosen)
                or contains_residual_pii(rejected)
            ):
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
                "id": hashlib.md5(
                    (
                        prompt + chosen
                    ).encode(
                        "utf-8"
                    )
                ).hexdigest(),
                "prompt":
                    prompt,
                "chosen":
                    chosen,
                "rejected":
                    rejected,
                "language":
                    language,
                "clinical_quality_score":
                    clinical_quality_score(
                        chosen
                    ),
                "safety_score":
                    safety_score(
                        rejected
                    ),
                "source":
                    item.get(
                        "source",
                        "unknown",
                    ),
                "metadata":
                    metadata,
            }

            preferences.append(
                preference_record
            )

    print(
        f"Skipped preference pairs due to residual PII: "
        f"{skipped_residual_pii}"
    )

    return preferences


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


def split_preferences(
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
        "\nBuilding DPO preferences..."
    )

    preferences = (
        build_preferences()
    )

    print(
        f"Generated "
        f"{len(preferences)} "
        f"preference pairs"
    )

    (
        train,
        validation,
        test,
        clinical_eval,
    ) = split_preferences(
        preferences
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
        "\nDPO dataset build completed"
    )


if __name__ == "__main__":
    main()
