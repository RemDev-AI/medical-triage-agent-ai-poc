# medical-triage-agent-ai-poc/backend/app/anonymization/validation.py

"""
Validation de conformité RGPD après anonymisation.

Support :
- Français (FR)
- Anglais (EN)

Vérifie qu'aucune entité PII détectable ne subsiste
après anonymisation.
"""

from __future__ import annotations

from app.anonymization.audit_logger import (
    audit_logger,
)
from app.anonymization.presidio_analyzer import (
    detect_language,
    detect_pii,
)

# ==========================================================
# PUBLIC API
# ==========================================================


def validate_no_pii(
    text: str,
    language: str | None = None,
) -> bool:
    """
    Vérifie qu'aucune donnée personnelle identifiable
    (PII) n'est encore présente dans le texte.

    Args:
        text:
            Texte à valider.

        language:
            "fr", "en" ou None pour auto-détection.

    Returns:
        True si aucune PII résiduelle n'est détectée.
    """

    if not text or not text.strip():

        audit_logger.info("Validation success | empty text")

        return True

    language = language or detect_language(text)

    findings = detect_pii(
        text=text,
        language=language,
    )

    if findings:

        audit_logger.warning(
            "Residual PII detected | "
            f"language={language} | "
            f"count={len(findings)}"
        )

        return False

    audit_logger.info(
        "Validation success | " f"language={language} | " "no residual PII"
    )

    return True


def contains_pii(
    text: str,
    language: str | None = None,
) -> bool:
    """
    Vérifie si le texte contient au moins une donnée
    personnelle identifiable (PII).

    Fonction complémentaire de `validate_no_pii` :
    contains_pii(text) == (not validate_no_pii(text))

    Args:
        text:
            Texte à analyser.

        language:
            "fr", "en" ou None pour auto-détection.

    Returns:
        True si au moins une PII est détectée dans le texte.
    """

    if not text or not text.strip():
        return False

    language = language or detect_language(text)

    findings = detect_pii(
        text=text,
        language=language,
    )

    return len(findings) > 0


# ==========================================================
# LOCAL TEST
# ==========================================================

if __name__ == "__main__":

    french_sample = """
    Bonjour,

    mon email est [REDACTED]

    mon téléphone est [REDACTED]

    MRN : [REDACTED]
    """

    english_sample = """
    Hello,

    my email is [REDACTED]

    my phone number is [REDACTED]

    MRN : [REDACTED]
    """

    print("\n=== FRENCH SAMPLE ===\n")

    print(validate_no_pii(french_sample))

    print("\n=== ENGLISH SAMPLE ===\n")

    print(validate_no_pii(english_sample))
