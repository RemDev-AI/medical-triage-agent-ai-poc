# medical-triage-agent-ai-poc/backend/app/anonymization/spacy_setup.py

"""
Configuration SpaCy multilingue (FR / EN).

Ce module centralise le chargement des modèles NLP utilisés
par Presidio pour la détection des entités sensibles (PII).
"""

from __future__ import annotations

from functools import lru_cache

import spacy
from spacy.language import Language

# ==========================================================
# SUPPORTED MODELS
# ==========================================================

SUPPORTED_MODELS: dict[str, str] = {
    "fr": "fr_core_news_md",
    "en": "en_core_web_md",
}

# ==========================================================
# PUBLIC API
# ==========================================================


def supported_languages() -> list[str]:
    """
    Retourne la liste des langues supportées.

    Returns:
        Liste des codes ISO.
    """

    return list(SUPPORTED_MODELS.keys())


def get_model_name(language: str) -> str:
    """
    Retourne le modèle SpaCy associé à une langue.

    Args:
        language:
            Code langue.

    Returns:
        Nom du modèle SpaCy.
    """

    if language not in SUPPORTED_MODELS:
        raise ValueError(
            f"Unsupported language '{language}'. "
            f"Supported languages: {', '.join(supported_languages())}"
        )

    return SUPPORTED_MODELS[language]


def get_presidio_configuration() -> dict:
    """
    Retourne la configuration Presidio compatible
    avec les modèles déclarés dans ce module.

    Returns:
        Configuration NlpEngineProvider.
    """

    return {
        "nlp_engine_name": "spacy",
        "models": [
            {
                "lang_code": language,
                "model_name": model_name,
            }
            for language, model_name in SUPPORTED_MODELS.items()
        ],
    }


@lru_cache(maxsize=None)
def load_spacy_model(
    language: str = "fr",
) -> Language:
    """
    Charge et met en cache un modèle SpaCy.

    Args:
        language:
            "fr" ou "en".

    Returns:
        Instance SpaCy Language.
    """

    model_name = get_model_name(language)

    try:
        return spacy.load(model_name)

    except OSError as exc:
        raise RuntimeError(
            f"SpaCy model '{model_name}' is not installed. "
            "Install project dependencies from requirements.txt."
        ) from exc


# ==========================================================
# LOCAL TEST
# ==========================================================

if __name__ == "__main__":

    print("\nSupported languages:")

    for lang in supported_languages():

        nlp = load_spacy_model(lang)

        print(f"[OK] Language={lang} " f"Model={get_model_name(lang)}")

        print(f"Pipeline: {nlp.pipe_names}")

    print("\nPresidio configuration:")

    print(get_presidio_configuration())
