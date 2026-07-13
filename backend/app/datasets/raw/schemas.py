# medical-triage-agent-ai-poc/backend/app/datasets/raw/schemas.py

"""
Schémas standardisés datasets médicaux.

Ces schémas servent de référence pour la normalisation des
datasets RAW avant la génération des jeux SFT et DPO.
"""

STANDARD_SCHEMA = {
    "id": str,
    "instruction": str,
    "response": str,
    "source": str,
    "language": str,
    "metadata": dict,
}

METADATA_SCHEMA = {
    "dataset_name": str,
    "dataset_subset": str,
    "medical_subject": str,
    "question_type": str,
    "symptoms": list,
    "medical_history": list,
    "vital_signs": dict,
    "confidence_score": float,
    "anonymized": bool,
    "split": str,
    "source_record_id": str,
}
