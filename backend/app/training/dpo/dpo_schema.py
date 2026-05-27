# medical-triage-agent-ai-poc/backend/app/training/dpo/dpo_schema.py

"""
Schéma dataset DPO médical.
"""

DPO_SCHEMA = {
    "id": str,
    "prompt": str,
    "chosen": str,
    "rejected": str,
    "clinical_quality_score": float,
    "safety_score": float,
    "source": str,
    "metadata": dict,
}
