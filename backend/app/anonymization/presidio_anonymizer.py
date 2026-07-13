# medical-triage-agent-ai-poc/backend/app/anonymization/presidio_anonymizer.py

"""
Anonymisation RGPD des datasets médicaux.

Support :
- Français (FR)
- Anglais (EN)

Compatible avec le pipeline d'anonymisation médical bilingue.

Mode recommandé pour SFT/DPO :
    PERSON -> [PERSON]
    EMAIL_ADDRESS -> [EMAIL]
    PHONE_NUMBER -> [PHONE]
    MEDICAL_RECORD_NUMBER -> [MEDICAL_RECORD_NUMBER]
"""

from __future__ import annotations

from presidio_anonymizer import AnonymizerEngine
from presidio_anonymizer.entities import OperatorConfig

from app.anonymization.audit_logger import (
    audit_logger,
)
from app.anonymization.presidio_analyzer import (
    detect_language,
    detect_pii,
)

# ==========================================================
# ANONYMIZER ENGINE
# ==========================================================

anonymizer = AnonymizerEngine()

# ==========================================================
# GENERIC STRATEGIES
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
}

# ==========================================================
# ENTITY-SPECIFIC REPLACEMENTS
# ==========================================================

ENTITY_REPLACEMENT_OPERATORS = {
    "PERSON": OperatorConfig(
        "replace",
        {"new_value": "[PERSON]"},
    ),
    "EMAIL_ADDRESS": OperatorConfig(
        "replace",
        {"new_value": "[EMAIL]"},
    ),
    "PHONE_NUMBER": OperatorConfig(
        "replace",
        {"new_value": "[PHONE]"},
    ),
    "LOCATION": OperatorConfig(
        "replace",
        {"new_value": "[LOCATION]"},
    ),
    "MEDICAL_RECORD_NUMBER": OperatorConfig(
        "replace",
        {"new_value": "[MEDICAL_RECORD_NUMBER]"},
    ),
    "PATIENT_ID": OperatorConfig(
        "replace",
        {"new_value": "[PATIENT_ID]"},
    ),
    "FRENCH_SOCIAL_SECURITY": OperatorConfig(
        "replace",
        {"new_value": "[FRENCH_SOCIAL_SECURITY]"},
    ),
    "US_SOCIAL_SECURITY": OperatorConfig(
        "replace",
        {"new_value": "[US_SOCIAL_SECURITY]"},
    ),
    "IP_ADDRESS": OperatorConfig(
        "replace",
        {"new_value": "[IP_ADDRESS]"},
    ),
    "URL": OperatorConfig(
        "replace",
        {"new_value": "[URL]"},
    ),
    "DEFAULT": OperatorConfig(
        "replace",
        {"new_value": "[REDACTED]"},
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
    Anonymise un texte contenant des données personnelles.

    Args:
        text:
            Texte source.

        strategy:
            replace | redact | mask

        language:
            fr | en | None

    Returns:
        Texte anonymisé.
    """

    if not text:
        return text

    if strategy not in {
        "replace",
        "redact",
        "mask",
    }:
        raise ValueError(f"Unknown strategy: {strategy}")

    language = language or detect_language(text)

    analyzer_results = detect_pii(
        text=text,
        language=language,
    )

    if not analyzer_results:

        audit_logger.info(
            "Anonymization skipped | " f"language={language} | " "findings=0"
        )

        return text

    if strategy == "replace":

        operators = ENTITY_REPLACEMENT_OPERATORS

    else:

        operators = {"DEFAULT": ANONYMIZATION_OPERATORS[strategy]}

    #    safe_results = []

    #    for result in analyzer_results:

    #        entity_text = text[
    #            result.start:result.end
    #        ]

    #        if (
    #            result.entity_type == "PERSON"
    #            and entity_text.lower() in MEDICAL_WHITELIST
    #        ):
    #            continue

    #        safe_results.append(result)

    #    analyzer_results = safe_results

    anonymized_result = anonymizer.anonymize(
        text=text,
        analyzer_results=analyzer_results,
        operators=operators,
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

    185067512345678

    Mon IP est 192.168.10.15

    Documentation :
    https://hopital.fr/patient/123
    """

    english_sample = """
    Hello,

    My name is John Smith.

    My email is john.smith@gmail.com

    My phone number is +1 555 123 4567

    MRN-458796

    123-45-6789

    Server:
    10.0.0.15

    Website:
    https://hospital.org/report
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
