# medical-triage-agent-ai-poc/backend/app/training/sft/sft_schema.py

"""
Schéma dataset SFT médical.
"""

SFT_SCHEMA = {
    "id": str,
    "instruction": str,
    "response": str,
    "source": str,
    "language": str,
    "confidence_score": float,
    "clinical_tags": list,
    "metadata": dict,
}
