# medical-triage-agent-ai-poc/backend/app/anonymization/spacy_setup.py

"""
Configuration SpaCy FR.
"""

from __future__ import annotations

import spacy

MODEL_NAME = "fr_core_news_md"


def load_spacy_model():
    """
    Charge le modèle SpaCy français.
    """

    try:
        return spacy.load(MODEL_NAME)

    except OSError as exc:
        raise RuntimeError(
            f"SpaCy model '{MODEL_NAME}' is not installed. "
            "Install project dependencies from requirements.txt."
        ) from exc


if __name__ == "__main__":

    nlp = load_spacy_model()

    print("SpaCy model loaded successfully")
    print(f"Pipeline: {nlp.pipe_names}")
