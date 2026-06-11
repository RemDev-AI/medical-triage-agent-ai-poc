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
import time
from pathlib import Path

from sklearn.model_selection import train_test_split

from backend.app.anonymization.presidio_anonymizer import anonymize_text

INPUT_DIR = Path("backend/app/datasets/processed/sft")
OUTPUT_DIR = Path("backend/app/datasets/processed/dpo")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

RANDOM_SEED = 42
random.seed(RANDOM_SEED)

EMAIL_PATTERN = re.compile(r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}")
PHONE_PATTERN = re.compile(r"(?:\+?\d[\d\s().-]{7,}\d)")


def contains_residual_pii(text: str) -> bool:
    if not text:
        return False
    return bool(
        EMAIL_PATTERN.search(text)
        or PHONE_PATTERN.search(text)
    )


def generate_rejected_response(response: str, language: str) -> str:
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

    for src, dst in replacements.items():
        degraded = degraded.replace(src, dst)
        degraded = degraded.replace(src.capitalize(), dst.capitalize())

    if degraded == response:
        sentences = response.split(". ")
        if len(sentences) > 1:
            degraded = ". ".join(sentences[: max(1, len(sentences)//2)])
        else:
            degraded = (
                "Informations limitées."
                if language == "fr"
                else "Limited information available."
            )

    return degraded.strip()


def clinical_quality_score(response: str) -> float:
    score = 0.80
    for term in [
        "doctor", "physician", "treatment", "diagnosis", "medical",
        "consult", "médecin", "traitement", "diagnostic",
        "consulter", "urgence", "emergency"
    ]:
        if term in response.lower():
            score += 0.02
    return min(score, 1.0)


def safety_score(response: str) -> float:
    score = 0.90
    for term in [
        "ignore", "search online", "wait and see", "guess",
        "internet", "ignorer", "automédication", "attendre"
    ]:
        if term in response.lower():
            score -= 0.10
    return max(score, 0.0)


def anonymize_preference_fields(prompt, chosen, rejected, language):
    return (
        anonymize_text(prompt, language=language),
        anonymize_text(chosen, language=language),
        anonymize_text(rejected, language=language),
    )


def build_preferences():
    sft_train = INPUT_DIR / "train.jsonl"

    preferences = []
    skipped_residual_pii = 0
    processed = 0

    rejected_time = 0.0
    anonymization_time = 0.0
    pii_time = 0.0

    slow_pairs = 0
    start = time.time()

    with open(sft_train, "r", encoding="utf-8") as f:
        for line in f:
            pair_start = time.time()
            processed += 1

            item = json.loads(line)

            language = item.get("language", "unknown")
            prompt = item["instruction"].strip()
            chosen = item["response"].strip()

            t0 = time.time()
            rejected = generate_rejected_response(chosen, language)
            rejected_time += time.time() - t0

            if rejected == chosen:
                continue

            t0 = time.time()
            prompt, chosen, rejected = anonymize_preference_fields(
                prompt, chosen, rejected, language
            )
            anonymization_time += time.time() - t0

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

            metadata = dict(item.get("metadata", {}))
            metadata["anonymized"] = True

            preferences.append({
                "id": hashlib.md5((prompt + chosen).encode("utf-8")).hexdigest(),  # noqa : E501
                "prompt": prompt,
                "chosen": chosen,
                "rejected": rejected,
                "language": language,
                "clinical_quality_score": clinical_quality_score(chosen),
                "safety_score": safety_score(rejected),
                "source": item.get("source", "unknown"),
                "metadata": metadata,
            })

            elapsed_pair = time.time() - pair_start

            if elapsed_pair > 0.5:
                slow_pairs += 1
                print(f"[SLOW] Pair {processed:,} processed in {elapsed_pair:.2f}s")  # noqa : E501

            if processed % 500 == 0:
                elapsed = time.time() - start
                print(
                    f"[PROGRESS] {processed:,} pairs | "
                    f"{elapsed:.1f}s | "
                    f"{elapsed/processed:.4f}s/pair"
                )

    total = time.time() - start

    print("\n" + "=" * 60)
    print("DPO PIPELINE SUMMARY")
    print("=" * 60)
    print(f"Pairs processed: {processed:,}")
    print(f"Preferences generated: {len(preferences):,}")
    print(f"Residual PII removed: {skipped_residual_pii:,}")
    print(f"Slow pairs (>0.5s): {slow_pairs:,}")
    print(f"generate_rejected_response(): {rejected_time:.2f}s")
    print(f"anonymize_preference_fields(): {anonymization_time:.2f}s")
    print(f"contains_residual_pii(): {pii_time:.2f}s")
    print(f"Average time per pair: {total/max(processed, 1):.4f}s")
    print(f"Total build time: {total:.2f}s")
    print("=" * 60)

    return preferences


def save_jsonl(records, path):
    with open(path, "w", encoding="utf-8") as f:
        for record in records:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")


def split_preferences(records):
    train, temp = train_test_split(
        records, test_size=0.20,
        random_state=RANDOM_SEED, shuffle=True
    )

    validation, test = train_test_split(
        temp, test_size=0.50,
        random_state=RANDOM_SEED, shuffle=True
    )

    clinical_eval = test[:min(100, len(test))]
    return train, validation, test, clinical_eval


def main():
    pipeline_start = time.time()

    print("\nBuilding DPO preferences...")

    preferences = build_preferences()

    train, validation, test, clinical_eval = split_preferences(preferences)

    save_jsonl(train, OUTPUT_DIR / "train.jsonl")
    save_jsonl(validation, OUTPUT_DIR / "validation.jsonl")
    save_jsonl(test, OUTPUT_DIR / "test.jsonl")
    save_jsonl(clinical_eval, OUTPUT_DIR / "clinical_eval.jsonl")

    total = time.time() - pipeline_start
    print(f"\nTotal pipeline time: {total:.2f}s")
    print("DPO dataset build completed")


if __name__ == "__main__":
    main()
