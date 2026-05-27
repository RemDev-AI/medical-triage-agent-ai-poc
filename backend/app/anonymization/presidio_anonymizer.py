# medical-triage-agent-ai-poc/backend/app/anonymization/presidio_anonymizer.py

"""
Anonymisation RGPD datasets médicaux.
"""

from __future__ import annotations

from presidio_anonymizer import AnonymizerEngine
from presidio_anonymizer.entities import OperatorConfig

from anonymization.presidio_analyzer import (
    detect_pii,
)

from anonymization.audit_logger import (
    audit_logger,
)

anonymizer = AnonymizerEngine()


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


def anonymize_text(
    text: str,
    strategy: str = "replace",
):
    """
    Anonymise un texte médical.
    """

    if strategy not in ANONYMIZATION_OPERATORS:
        raise ValueError(
            f"Unknown strategy: {strategy}"
        )

    analyzer_results = detect_pii(text)

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
        f"Anonymization applied | strategy={strategy}"
    )

    return anonymized_result.text


if __name__ == "__main__":

    text = """
    Patient Jean Dupont
    Email : jean@gmail.com
    MRN-458796
    """

    output = anonymize_text(
        text=text,
        strategy="replace",
    )

    print(output)
