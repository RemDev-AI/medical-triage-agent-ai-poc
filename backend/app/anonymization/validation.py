# medical-triage-agent-ai-poc/backend/app/anonymization/validation.py

"""
Validation conformité anonymisation RGPD.
"""

from __future__ import annotations

from anonymization.presidio_analyzer import (
    detect_pii,
)

from anonymization.audit_logger import (
    audit_logger,
)


def validate_no_pii(text: str) -> bool:
    """
    Vérifie qu'aucun PII n'est encore présent.
    """

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
    """

    is_valid = validate_no_pii(sample)

    print(f"Validation: {is_valid}")
