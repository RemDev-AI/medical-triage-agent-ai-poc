# medical-triage-agent-ai-poc/backend/app/datasets/raw/standardize_datasets.py

"""
Pipeline de standardisation des datasets médicaux.

Convertit les datasets RAW vers un schéma commun
utilisable pour :

- SFT
- DPO
- anonymisation RGPD
- évaluations cliniques
- RAG médical
"""

from pathlib import Path
import json

INPUT_DIR = Path("backend/app/datasets/raw/data")

OUTPUT_DIR = Path("backend/app/datasets/raw/standardized")

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True,
)


def safe_str(value) -> str:
    """
    Convertit les valeurs None en chaîne vide
    et applique strip().
    """

    if value is None:
        return ""

    return str(value).strip()


def detect_language(
    source_name: str,
) -> str:
    """
    Détermine la langue du dataset.
    """

    if source_name.startswith("mediqa"):
        return "fr"

    return "en"


def extract_id(
    record: dict,
) -> str:
    """
    Extraction robuste des identifiants.
    """

    return str(
        record.get("id")
        or record.get("question_id")
        or record.get("prompt_id")
        or record.get("document_id")
        or ""
    )


def build_mcq_instruction(
    record: dict,
) -> str:
    """
    Construit l'instruction pour
    les datasets MCQU / MCQM.
    """

    clinical_case = safe_str(record.get("clinical_case"))

    question = safe_str(record.get("question"))

    answers = []

    for letter in [
        "a",
        "b",
        "c",
        "d",
        "e",
    ]:
        value = safe_str(record.get(f"answer_{letter}"))

        if value:
            answers.append(f"{letter.upper()}. " f"{value}")

    parts = []

    if clinical_case:
        parts.append(clinical_case)

    if question:
        parts.append(question)

    if answers:
        parts.append("\n".join(answers))

    return "\n\n".join(parts)


def build_mcq_response(
    record: dict,
) -> str:
    """
    Génère la réponse cible
    pour les QCU / QCM.
    """

    correct_answers = safe_str(record.get("correct_answers"))

    if not correct_answers:
        return ""

    # Détection robuste
    # des réponses multiples.
    if "," in correct_answers:
        return "Réponses correctes : " f"{correct_answers}"

    return "Réponse correcte : " f"{correct_answers}"


def get_confidence_score(
    source_name: str,
) -> float:
    """
    Score de confiance initial.
    """

    if source_name.startswith("mediqa"):
        return 0.95

    if source_name == "medquad":
        return 0.90

    return 1.0


def normalize_preference_score(
    score,
) -> float:
    """
    Normalise les scores DPO
    vers l'intervalle [0,1].

    UltraMedical utilise
    généralement une échelle
    de 1 à 5.
    """

    try:
        score = float(score)
    except (
        TypeError,
        ValueError,
    ):
        return 0.0

    return max(
        0.0,
        min(
            score / 5.0,
            1.0,
        ),
    )


def standardize_ultramedical(
    record: dict,
    source_name: str,
):
    """
    Standardisation du dataset
    UltraMedical Preference.
    """

    chosen = record.get(
        "chosen",
        [],
    )

    rejected = record.get(
        "rejected",
        [],
    )

    chosen_response = ""
    rejected_response = ""

    if (
        isinstance(
            chosen,
            list,
        )
        and len(chosen) >= 2
        and isinstance(
            chosen[1],
            dict,
        )
    ):
        chosen_response = safe_str(chosen[1].get("content"))

    if (
        isinstance(
            rejected,
            list,
        )
        and len(rejected) >= 2
        and isinstance(
            rejected[1],
            dict,
        )
    ):
        rejected_response = safe_str(rejected[1].get("content"))

    metadata = record.get(
        "metadata",
        {},
    )

    raw_score = 0.0

    if isinstance(
        metadata,
        dict,
    ):
        raw_score = metadata.get(
            "chosen",
            {},
        ).get(
            "score",
            0.0,
        )

    confidence_score = normalize_preference_score(raw_score)

    return {
        "id": extract_id(record),
        "instruction": safe_str(record.get("prompt")),
        "response": chosen_response,
        "source": source_name,
        "language": "en",
        "metadata": {
            "dataset_name": source_name,
            "dataset_subset": "",
            "medical_subject": "",
            "question_type": "",
            "symptoms": [],
            "medical_history": [],
            "vital_signs": {},
            "confidence_score": confidence_score,
            "anonymized": False,
            "split": "",
            "source_record_id": (extract_id(record)),
            "rejected_response": (rejected_response),
            "original_fields": list(record.keys()),
        },
    }


def standardize_mcq(
    record: dict,
    source_name: str,
):
    """
    Standardisation
    MediQA MCQU / MCQM.
    """

    return {
        "id": extract_id(record),
        "instruction": (build_mcq_instruction(record)),
        "response": (build_mcq_response(record)),
        "source": source_name,
        "language": "fr",
        "metadata": {
            "dataset_name": source_name,
            "dataset_subset": safe_str(record.get("task")),
            "medical_subject": safe_str(record.get("medical_subject")),
            "question_type": safe_str(record.get("question_type")),
            "symptoms": [],
            "medical_history": [],
            "vital_signs": {},
            "confidence_score": 0.95,
            "anonymized": False,
            "split": "",
            "source_record_id": (extract_id(record)),
            "correct_answers": safe_str(record.get("correct_answers")),
            "original_fields": list(record.keys()),
        },
    }


def standardize_generic(
    record: dict,
    source_name: str,
):
    """
    Standardisation générique :

    - MediQA OEQ
    - MedQuAD
    """

    instruction = safe_str(record.get("question") or record.get("instruction"))

    response = safe_str(record.get("answer") or record.get("response"))

    medical_subject = safe_str(record.get("medical_subject") or record.get("category"))

    question_type = safe_str(record.get("question_type"))

    metadata = {
        "dataset_name": source_name,
        "dataset_subset": "",
        "medical_subject": (medical_subject),
        "question_type": (question_type),
        "symptoms": [],
        "medical_history": [],
        "vital_signs": {},
        "confidence_score": (get_confidence_score(source_name)),
        "anonymized": False,
        "split": "",
        "source_record_id": (extract_id(record)),
        "original_fields": list(record.keys()),
    }

    # Enrichissement spécifique
    # MedQuAD pour RAG médical.
    if source_name == "medquad":
        metadata.update(
            {
                "question_focus": (safe_str(record.get("question_focus"))),
                "umls_cui": safe_str(record.get("umls_cui")),
                "umls_semantic_group": (safe_str(record.get("umls_semantic_group"))),
                "document_source": (safe_str(record.get("document_source"))),
            }
        )

    return {
        "id": extract_id(record),
        "instruction": instruction,
        "response": response,
        "source": source_name,
        "language": (detect_language(source_name)),
        "metadata": metadata,
    }


def standardize_record(
    record: dict,
    source_name: str,
):
    """
    Routeur principal.
    """

    if source_name == ("ultramedical_preference"):
        return standardize_ultramedical(
            record,
            source_name,
        )

    if source_name in (
        "mediqa_mcqu",
        "mediqa_mcqm",
    ):
        return standardize_mcq(
            record,
            source_name,
        )

    return standardize_generic(
        record,
        source_name,
    )


def process_file(
    file_path: Path,
):
    """
    Standardise un fichier JSONL.
    """

    output_path = OUTPUT_DIR / file_path.name

    total_records = 0
    error_count = 0

    with (
        open(
            file_path,
            "r",
            encoding="utf-8",
        ) as infile,
        open(
            output_path,
            "w",
            encoding="utf-8",
        ) as outfile,
    ):

        for line_number, line in enumerate(
            infile,
            start=1,
        ):
            line = line.strip()

            if not line:
                continue

            try:
                record = json.loads(line)

                standardized = standardize_record(
                    record,
                    file_path.stem,
                )

                outfile.write(
                    json.dumps(
                        standardized,
                        ensure_ascii=False,
                    )
                    + "\n"
                )

                total_records += 1

            except Exception as exc:
                error_count += 1

                print(f"[ERROR] " f"{file_path.name}" f":{line_number} " f"{exc}")

    print(f"Standardized -> " f"{output_path} " f"({total_records} records)")

    if error_count:
        print(f"Errors -> " f"{error_count}")


def main():
    """
    Point d'entrée principal.
    """

    files = sorted(INPUT_DIR.glob("*.jsonl"))

    if not files:
        print("No JSONL files found.")
        return

    for file_path in files:
        process_file(file_path)


if __name__ == "__main__":
    main()
