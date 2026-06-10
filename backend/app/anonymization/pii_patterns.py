# medical-triage-agent-ai-poc/backend/app/anonymization/pii_patterns.py

"""
Patterns PII médicaux personnalisés.

Support :
- Français (FR)
- Anglais (EN)

Utilisé par Presidio pour détecter les identifiants
médicaux et administratifs spécifiques au domaine santé.
"""

from __future__ import annotations

# ==========================================================
# LANGUAGE-SPECIFIC PATTERNS
# ==========================================================

LANGUAGE_PATTERNS = {
    "fr": [
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
    ],
    "en": [
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
            "name": "US_SOCIAL_SECURITY",
            "regex": r"\b\d{3}-\d{2}-\d{4}\b",
            "score": 0.95,
        },
    ],
}

# ==========================================================
# BACKWARD COMPATIBILITY
# ==========================================================

MEDICAL_PII_PATTERNS = (
    LANGUAGE_PATTERNS["fr"]
    + LANGUAGE_PATTERNS["en"]
)

# ==========================================================
# PUBLIC API
# ==========================================================


def get_pii_patterns(
    language: str,
) -> list[dict]:
    """
    Retourne les patterns adaptés à une langue.

    Args:
        language:
            "fr" ou "en"

    Returns:
        Liste de patterns Presidio.
    """

    return LANGUAGE_PATTERNS.get(
        language,
        [],
    )


def supported_languages() -> list[str]:
    """
    Retourne les langues supportées.
    """

    return list(
        LANGUAGE_PATTERNS.keys()
    )


# ==========================================================
# LOCAL TEST
# ==========================================================

if __name__ == "__main__":

    for language in supported_languages():

        patterns = get_pii_patterns(
            language
        )

        print(
            f"\nLanguage={language}"
        )

        for pattern in patterns:

            print(
                f" - {pattern['name']}"
            )
