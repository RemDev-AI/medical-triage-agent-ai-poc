# medical-triage-agent-ai-poc/backend/app/anonymization/validation.py

"""
Validation conformité anonymisation RGPD.
"""

from __future__ import annotations

from backend.app.anonymization.presidio_analyzer import (
    detect_pii,
)

from backend.app.anonymization.audit_logger import (
    audit_logger,
)


def validate_no_pii(text: str) -> bool:
    """
    Vérifie qu'aucune donnée personnelle identifiable
    (PII) n'est encore présente dans le texte.
    """

    if not text or not text.strip():

        audit_logger.info(
            "Validation success | empty text"
        )

        return True

    findings = detect_pii(text)

    if findings:

        audit_logger.warning(
            f"Residual PII detected | count={len(findings)}"
        )

        return False

    audit_logger.info(
        "Validation success | no residual PII"
    )

    return True


if __name__ == "__main__":

    sample = """
    Bonjour,

    mon email est [REDACTED]

    mon téléphone est [REDACTED]

    MRN : [REDACTED]
    """

    is_valid = validate_no_pii(sample)

    print(f"Validation: {is_valid}")
