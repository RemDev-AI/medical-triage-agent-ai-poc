# medical-triage-agent-ai-poc/backend/app/tests/unit/test_anonymization.py

import pytest  # noqa : F401


def test_email_masking():
    """
    Email address should be anonymized.
    """

    from backend.app.anonymization.validation import contains_pii

    text = "Contactez moi à john.doe@email.com"

    result = contains_pii(text)

    assert result is True


def test_phone_masking():
    """
    Phone number should be detected.
    """

    from backend.app.anonymization.validation import contains_pii

    text = "Mon numéro est 0612345678"

    result = contains_pii(text)

    assert result is True


def test_patient_name_detection():
    """
    Patient name should be detected.
    """

    from backend.app.anonymization.validation import contains_pii

    text = "Le patient Jean Dupont présente une toux."

    result = contains_pii(text)

    assert result is True


def test_clean_medical_text():
    """
    Medical text without PII should pass validation.
    """

    from backend.app.anonymization.validation import contains_pii

    text = (
        "Patient présentant une fièvre "
        "depuis trois jours."
    )

    result = contains_pii(text)

    assert result is False


def test_anonymizer_returns_string():
    """
    Anonymizer should always return a string.
    """

    from backend.app.anonymization.presidio_anonymizer import (
        anonymize_text,
    )

    text = "Jean Dupont"

    result = anonymize_text(text)

    assert isinstance(result, str)
