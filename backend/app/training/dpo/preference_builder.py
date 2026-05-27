# medical-triage-agent-ai-poc/backend/app/training/dpo/preference_builder.py

"""
Construction dataset DPO médical.
"""

from __future__ import annotations

import hashlib
import json
import random
from pathlib import Path

from sklearn.model_selection import train_test_split

INPUT_DIR = Path(
    "backend/app/datasets/processed/sft"
)

OUTPUT_DIR = Path(
    "backend/app/datasets/processed/dpo"
)

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

random.seed(42)


def generate_rejected_response(response: str):

    degraded = (
        response
        .replace("urgence", "attendre")
        .replace("médecin", "internet")
        .replace("consultation", "automédication")
    )

    return degraded


def clinical_quality_score(chosen: str):

    score = 0.85

    if "urgence" in chosen.lower():
        score += 0.05

    if "consulter" in chosen.lower():
        score += 0.05

    return min(score, 1.0)


def safety_score(response: str):

    score = 0.90

    dangerous_terms = [
        "ignorer",
        "automédication",
        "attendre plusieurs jours",
    ]

    if any(
        term in response.lower()
        for term in dangerous_terms
    ):
        score -= 0.25

    return max(score, 0.0)


def build_preferences():

    sft_train = (
        INPUT_DIR / "train.jsonl"
    )

    preferences = []

    with open(
        sft_train,
        "r",
        encoding="utf-8",
    ) as f:

        for line in f:

            item = json.loads(line)

            chosen = item["response"]

            rejected = generate_rejected_response(
                chosen
            )

            preference_record = {
                "id": hashlib.md5(
                    chosen.encode("utf-8")
                ).hexdigest(),
                "prompt": item["instruction"],
                "chosen": chosen,
                "rejected": rejected,
                "clinical_quality_score":
                    clinical_quality_score(
                        chosen
                    ),
                "safety_score":
                    safety_score(rejected),
                "source": item["source"],
                "metadata": item["metadata"],
            }

            preferences.append(
                preference_record
            )

    return preferences


def save_jsonl(records, path):

    with open(path, "w", encoding="utf-8") as f:

        for record in records:
            f.write(
                json.dumps(
                    record,
                    ensure_ascii=False,
                ) + "\n"
            )


def split_preferences(records):

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

    print("\nBuilding DPO preferences...")

    preferences = build_preferences()

    (
        train,
        validation,
        test,
        clinical_eval,
    ) = split_preferences(preferences)

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

    print("\nDPO dataset build completed")


if __name__ == "__main__":
    main()
