# medical-triage-agent-ai-poc/backend/app/anonymization/presidio_analyzer.py

"""
Analyse PII avec Microsoft Presidio.
"""

from __future__ import annotations

from presidio_analyzer import (
    AnalyzerEngine,
    Pattern,
    PatternRecognizer,
)

from anonymization.spacy_setup import (
    load_spacy_model,
)

from anonymization.pii_patterns import (
    MEDICAL_PII_PATTERNS,
)

from anonymization.audit_logger import (
    audit_logger,
)

nlp = load_spacy_model()

analyzer = AnalyzerEngine(
    supported_languages=["fr"],
)

for pii_pattern in MEDICAL_PII_PATTERNS:

    recognizer = PatternRecognizer(
        supported_entity=pii_pattern["name"],
        patterns=[
            Pattern(
                name=pii_pattern["name"],
                regex=pii_pattern["regex"],
                score=pii_pattern["score"],
            )
        ],
        supported_language="fr",
    )

    analyzer.registry.add_recognizer(recognizer)


DEFAULT_ENTITIES = [
    "PERSON",
    "EMAIL_ADDRESS",
    "PHONE_NUMBER",
    "LOCATION",
    "MEDICAL_RECORD_NUMBER",
    "PATIENT_ID",
    "FRENCH_SOCIAL_SECURITY",
]


def detect_pii(
    text: str,
    language: str = "fr",
):
    """
    Détection des entités PII.
    """

    results = analyzer.analyze(
        text=text,
        entities=DEFAULT_ENTITIES,
        language=language,
    )

    audit_logger.info(
        f"PII detection executed | findings={len(results)}"
    )

    return results


if __name__ == "__main__":

    sample = """
    Bonjour,
    Je suis Jean Dupont.
    Mon email est jean.dupont@gmail.com
    MRN-458796
    """

    detections = detect_pii(sample)

    for item in detections:
        print(item)
