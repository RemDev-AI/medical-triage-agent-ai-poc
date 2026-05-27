# medical-triage-agent-ai-poc/backend/app/datasets/raw/schemas.py

"""
Schémas standardisés datasets médicaux.
"""

STANDARD_SCHEMA = {
    "id": str,
    "instruction": str,
    "response": str,
    "source": str,
    "language": str,
    "metadata": dict,
}
