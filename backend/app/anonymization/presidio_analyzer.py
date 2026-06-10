# medical-triage-agent-ai-poc/backend/app/anonymization/presidio_analyzer.py

"""
Analyse PII avec Microsoft Presidio.

Support :
- Français (FR)
- Anglais (EN)

Compatible avec le pipeline d'anonymisation médical bilingue.
"""

from __future__ import annotations

from langdetect import LangDetectException
from langdetect import detect

from presidio_analyzer import (
    AnalyzerEngine,
    Pattern,
    PatternRecognizer,
    RecognizerResult,
)
from presidio_analyzer.nlp_engine import (
    NlpEngineProvider,
)

from backend.app.anonymization.audit_logger import (
    audit_logger,
)
from backend.app.anonymization.pii_patterns import (
    get_pii_patterns,
    supported_languages,
)
from backend.app.anonymization.spacy_setup import (
    get_presidio_configuration,
)

# ==========================================================
# NLP ENGINE CONFIGURATION
# ==========================================================

provider = NlpEngineProvider(
    nlp_configuration=get_presidio_configuration()
)

nlp_engine = provider.create_engine()

# ==========================================================
# ANALYZER
# ==========================================================

SUPPORTED_LANGUAGES = supported_languages()

analyzer = AnalyzerEngine(
    nlp_engine=nlp_engine,
    supported_languages=SUPPORTED_LANGUAGES,
)

# ==========================================================
# CUSTOM MEDICAL PATTERNS
# ==========================================================

for language in SUPPORTED_LANGUAGES:

    for pii_pattern in get_pii_patterns(language):

        recognizer = PatternRecognizer(
            supported_entity=pii_pattern["name"],
            patterns=[
                Pattern(
                    name=pii_pattern["name"],
                    regex=pii_pattern["regex"],
                    score=pii_pattern["score"],
                )
            ],
            supported_language=language,
        )

        analyzer.registry.add_recognizer(
            recognizer
        )

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
    "US_SOCIAL_SECURITY",
]

# ==========================================================
# LANGUAGE DETECTION
# ==========================================================


def detect_language(text: str) -> str:
    """
    Détection automatique de langue.

    Args:
        text:
            Texte à analyser.

    Returns:
        "fr" ou "en"
    """

    if not text or not text.strip():
        return "en"

    try:
        language = detect(text)

    except LangDetectException:
        return "en"

    if language not in SUPPORTED_LANGUAGES:
        return "en"

    return language


# ==========================================================
# PUBLIC API
# ==========================================================


def detect_pii(
    text: str,
    language: str | None = None,
) -> list[RecognizerResult]:
    """
    Détection des entités PII.

    Args:
        text:
            Texte à analyser.

        language:
            "fr", "en" ou None pour auto-détection.

    Returns:
        Liste des entités détectées.
    """

    if not text or not text.strip():
        return []

    language = language or detect_language(text)

    results = analyzer.analyze(
        text=text,
        entities=DEFAULT_ENTITIES,
        language=language,
    )

    audit_logger.info(
        "PII detection executed | "
        f"language={language} | "
        f"findings={len(results)}"
    )

    return results


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
    """

    english_sample = """
    Hello,

    My name is John Smith.

    My email is john.smith@gmail.com

    My phone number is +1 555 123 4567

    MRN-458796

    123-45-6789
    """

    print("\n=== FRENCH SAMPLE ===\n")

    for item in detect_pii(french_sample):
        print(item)

    print("\n=== ENGLISH SAMPLE ===\n")

    for item in detect_pii(english_sample):
        print(item)
