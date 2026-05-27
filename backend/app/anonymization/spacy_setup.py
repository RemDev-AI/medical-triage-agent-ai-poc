# medical-triage-agent-ai-poc/backend/app/anonymization/spacy_setup.py

"""
Configuration SpaCy FR.
"""

from __future__ import annotations

import spacy
from spacy.cli import download

MODEL_NAME = "fr_core_news_md"


def load_spacy_model():
    """
    Charge le modèle SpaCy français.
    """

    try:
        nlp = spacy.load(MODEL_NAME)

    except OSError:
        print(f"Downloading SpaCy model: {MODEL_NAME}")
        download(MODEL_NAME)
        nlp = spacy.load(MODEL_NAME)

    return nlp


if __name__ == "__main__":

    nlp = load_spacy_model()

    print("SpaCy model loaded successfully")
    print(f"Pipeline: {nlp.pipe_names}")
