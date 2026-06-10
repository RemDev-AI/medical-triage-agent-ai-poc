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

from presidio_analyzer.nlp_engine import (
    NlpEngineProvider,
)

from backend.app.anonymization.pii_patterns import (
    MEDICAL_PII_PATTERNS,
)

from backend.app.anonymization.audit_logger import (
    audit_logger,
)

# ==========================================================
# NLP ENGINE CONFIGURATION
# ==========================================================

NLP_CONFIGURATION = {
    "nlp_engine_name": "spacy",
    "models": [
        {
            "lang_code": "fr",
            "model_name": "fr_core_news_md",
        }
    ],
}

provider = NlpEngineProvider(
    nlp_configuration=NLP_CONFIGURATION
)

nlp_engine = provider.create_engine()

# ==========================================================
# ANALYZER
# ==========================================================

analyzer = AnalyzerEngine(
    nlp_engine=nlp_engine,
    supported_languages=["fr"],
)

# ==========================================================
# CUSTOM MEDICAL PATTERNS
# ==========================================================

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

# ==========================================================
# DEFAULT ENTITIES
# ==========================================================

DEFAULT_ENTITIES = [
    "PERSON",
    "EMAIL_ADDRESS",
    "PHONE_NUMBER",
    "LOCATION",
    "MEDICAL_RECORD_NUMBER",
    "PATIENT_ID",
    "FRENCH_SOCIAL_SECURITY",
]

# ==========================================================
# PUBLIC API
# ==========================================================


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


# ==========================================================
# LOCAL TEST
# ==========================================================

if __name__ == "__main__":

    sample = """
    Bonjour,

    Je suis Jean Dupont.

    Mon email est jean.dupont@gmail.com

    Mon téléphone est 06 12 34 56 78

    MRN-458796
    """

    detections = detect_pii(sample)

    print("\nDetected entities:\n")

    for item in detections:
        print(item)
