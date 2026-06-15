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
# CUSTOM PHONE RECOGNIZER
# ==========================================================

PHONE_PATTERNS = [
    Pattern(
        name="fr_phone",
        regex=(
            r"(?<!\d)"
            r"(?:0[1-9])"
            r"(?:[\s.-]?\d{2}){4}"
            r"(?!\d)"
        ),
        score=0.90,
    ),
    Pattern(
        name="international_phone",
        regex=(
            r"(?<!\d)"
            r"\+\d{1,3}"
            r"(?:[\s.-]?\d{1,4}){2,6}"
            r"(?!\d)"
        ),
        score=0.90,
    ),
    Pattern(
        name="us_phone",
        regex=(
            r"(?<!\d)"
            r"(?:\(\d{3}\)|\d{3})"
            r"[\s.-]?"
            r"\d{3}"
            r"[\s.-]?"
            r"\d{4}"
            r"(?!\d)"
        ),
        score=0.90,
    ),
]

for language in SUPPORTED_LANGUAGES:

    phone_recognizer = PatternRecognizer(
        supported_entity="PHONE_NUMBER",
        patterns=PHONE_PATTERNS,
        supported_language=language,
    )

    analyzer.registry.add_recognizer(
        phone_recognizer
    )

# ==========================================================
# ENTITY PRIORITY
# ==========================================================

ENTITY_PRIORITY = {
    "EMAIL_ADDRESS": 100,
    "FRENCH_SOCIAL_SECURITY": 99,
    "US_SOCIAL_SECURITY": 99,
    "MEDICAL_RECORD_NUMBER": 95,
    "PATIENT_ID": 90,
    "IP_ADDRESS": 88,
    "URL": 87,
    "PHONE_NUMBER": 85,
    "PERSON": 50,
    # "LOCATION": 40,
}


# ==========================================================
# DEFAULT ENTITIES
# ==========================================================

DEFAULT_ENTITIES = [
    "PERSON",
    "EMAIL_ADDRESS",
    "PHONE_NUMBER",
    # "LOCATION",
    "MEDICAL_RECORD_NUMBER",
    "PATIENT_ID",
    "FRENCH_SOCIAL_SECURITY",
    "US_SOCIAL_SECURITY",
    "IP_ADDRESS",
    "URL",
]

MEDICAL_WHITELIST = {
    "Accès",
    "accès",
    "Paludisme",
    "paludisme",
    "Primo-invasion",
    "primo-invasion",
    "Pernicieux",
    "pernicieux",
    "Hodgkin",
    "Crohn",
    "Parkinson",
    "Alzheimer",
    "Boston criteria",
    "Paris classification",
    "Lyon score",
}

# ==========================================================
# LANGUAGE DETECTION
# ==========================================================


def detect_language(text: str) -> str:
    """
    Détection automatique de langue.
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
# OVERLAP RESOLUTION
# ==========================================================


def _entities_overlap(
    entity_a: RecognizerResult,
    entity_b: RecognizerResult,
) -> bool:
    """
    Détecte tout recouvrement partiel ou total.
    """

    return (
        entity_a.start < entity_b.end
        and entity_b.start < entity_a.end
    )


def _resolve_overlaps(
    results: list[RecognizerResult],
) -> list[RecognizerResult]:
    """
    Conserve uniquement l'entité la plus prioritaire
    lorsqu'il existe un chevauchement.
    """

    sorted_results = sorted(
        results,
        key=lambda item: (
            ENTITY_PRIORITY.get(
                item.entity_type,
                0,
            ),
            item.score,
        ),
        reverse=True,
    )

    filtered: list[RecognizerResult] = []

    for candidate in sorted_results:

        has_conflict = any(
            _entities_overlap(
                candidate,
                existing,
            )
            for existing in filtered
        )

        if not has_conflict:
            filtered.append(candidate)

    return sorted(
        filtered,
        key=lambda item: item.start,
    )


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
            "fr", "en" ou None.

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
        score_threshold=0.85,
    )

    results = _resolve_overlaps(results)

    audit_logger.info(
        "PII detection executed | "
        f"language={language} | "
        f"findings={len(results)}"
    )
    
    filtered_results = []

    for result in results:

        entity_text = text[
            result.start:result.end
        ]

        if entity_text in MEDICAL_WHITELIST:
            continue

        filtered_results.append(result)

    results = filtered_results

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
