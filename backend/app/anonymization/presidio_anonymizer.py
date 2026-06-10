# medical-triage-agent-ai-poc/backend/app/anonymization/presidio_anonymizer.py

"""
Anonymisation RGPD datasets médicaux.
"""

from __future__ import annotations

from presidio_anonymizer import AnonymizerEngine
from presidio_anonymizer.entities import OperatorConfig

from backend.app.anonymization.presidio_analyzer import (
    detect_pii,
)

from backend.app.anonymization.audit_logger import (
    audit_logger,
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
) -> str:
    """
    Anonymise un texte médical contenant des PII.
    """

    if not text:
        return text

    if strategy not in ANONYMIZATION_OPERATORS:
        raise ValueError(
            f"Unknown strategy: {strategy}"
        )

    analyzer_results = detect_pii(text)

    if not analyzer_results:

        audit_logger.info(
            "Anonymization skipped | findings=0"
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
        f"strategy={strategy} | "
        f"findings={len(analyzer_results)}"
    )

    return anonymized_result.text


# ==========================================================
# LOCAL TEST
# ==========================================================

if __name__ == "__main__":

    sample = """
    Patient Jean Dupont

    Email : jean@gmail.com

    Téléphone : 06 12 34 56 78

    MRN-458796
    """

    output = anonymize_text(
        text=sample,
        strategy="replace",
    )

    print("\nOriginal:\n")
    print(sample)

    print("\nAnonymized:\n")
    print(output)
