# medical-triage-agent-ai-poc/backend/app/anonymization/pii_patterns.py

"""
Patterns PII médicaux personnalisés.
"""

MEDICAL_PII_PATTERNS = [
    {
        "name": "MEDICAL_RECORD_NUMBER",
        "regex": r"\bMRN[- ]?\d{5,12}\b",
        "score": 0.85,
    },
    {
        "name": "PATIENT_ID",
        "regex": r"\bPAT[- ]?\d{4,10}\b",
        "score": 0.80,
    },
    {
        "name": "FRENCH_SOCIAL_SECURITY",
        "regex": r"\b[12]\d{14}\b",
        "score": 0.95,
    },
]
