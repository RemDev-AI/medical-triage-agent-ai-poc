# medical-triage-agent-ai-poc/backend/app/anonymization/presidio_anonymizer.py

"""
Anonymisation RGPD des datasets médicaux.

Support :
- Français (FR)
- Anglais (EN)

Compatible avec le pipeline d'anonymisation médical bilingue.
"""

from __future__ import annotations

from presidio_anonymizer import AnonymizerEngine
from presidio_anonymizer.entities import OperatorConfig

from backend.app.anonymization.audit_logger import (
    audit_logger,
)
from backend.app.anonymization.presidio_analyzer import (
    detect_language,
    detect_pii,
)

# ==========================================================
# ANONYMIZER ENGINE
# ==========================================================

anonymizer = AnonymizerEngine()

# ==========================================================
# ANONYMIZATION STRATEGIES
# ==========================================================

ANONYMIZATION_OPERATORS = {
    "mask": OperatorConfig(
        "mask",
        {
            "masking_char": "*",
            "chars_to_mask": 100,
            "from_end": True,
        },
    ),
    "redact": OperatorConfig(
        "redact",
        {},
    ),
    "replace": OperatorConfig(
        "replace",
        {
            "new_value": "[REDACTED]",
        },
    ),
}

# ==========================================================
# PUBLIC API
# ==========================================================


def anonymize_text(
    text: str,
    strategy: str = "replace",
    language: str | None = None,
) -> str:
    """
    Anonymise un texte médical contenant des PII.

    Args:
        text:
            Texte source.

        strategy:
            Stratégie d'anonymisation :
            - replace
            - redact
            - mask

        language:
            "fr", "en" ou None pour auto-détection.

    Returns:
        Texte anonymisé.
    """

    if not text:
        return text

    if strategy not in ANONYMIZATION_OPERATORS:
        raise ValueError(
            f"Unknown strategy: {strategy}"
        )

    language = language or detect_language(text)

    analyzer_results = detect_pii(
        text=text,
        language=language,
    )

    if not analyzer_results:

        audit_logger.info(
            "Anonymization skipped | "
            f"language={language} | "
            "findings=0"
        )

        return text

    anonymized_result = anonymizer.anonymize(
        text=text,
        analyzer_results=analyzer_results,
        operators={
            "DEFAULT": ANONYMIZATION_OPERATORS[
                strategy
            ]
        },
    )

    audit_logger.info(
        "Anonymization applied | "
        f"language={language} | "
        f"strategy={strategy} | "
        f"findings={len(analyzer_results)}"
    )

    return anonymized_result.text


# ==========================================================
# LOCAL TEST
# ==========================================================

if __name__ == "__main__":

    french_sample = """
    Bonjour,

    Je suis Jean Dupont.

    Mon email est jean.dupont@gmail.com

    Mon téléphone est 06 12 34 56 78

    MRN-458796
    """

    english_sample = """
    Hello,

    My name is John Smith.

    My email is john.smith@gmail.com

    My phone number is +1 555 123 4567

    MRN-458796
    """

    print("\n=== FRENCH SAMPLE ===\n")

    print(
        anonymize_text(
            text=french_sample,
            strategy="replace",
        )
    )

    print("\n=== ENGLISH SAMPLE ===\n")

    print(
        anonymize_text(
            text=english_sample,
            strategy="replace",
        )
    )
