# medical-triage-agent-ai-poc/backend/app/llm/loaders/tokenizer_loader.py

"""
Tokenizer loader for Qwen models.

Features:
- Hugging Face tokenizer loading
- automatic pad token configuration
- inference-ready setup
"""

from __future__ import annotations

import logging

from transformers import (
    AutoTokenizer,
    PreTrainedTokenizerBase,
)

logger = logging.getLogger(__name__)

# Aligné avec DEFAULT_BASE_MODEL_NAME de model_loader.py, pour que
# tokenizer et modèle par défaut restent cohérents entre eux.
DEFAULT_MODEL_NAME = "Qwen/Qwen3-1.7B-Base"


def load_tokenizer(
    model_name: str = DEFAULT_MODEL_NAME,
    revision: str = "main",
) -> PreTrainedTokenizerBase:
    """
    Load tokenizer from Hugging Face.

    Args:
        model_name:
            Hugging Face model identifier.
        revision:
            Hub revision (commit SHA, tag, or branch) to pin, to avoid
            loading unexpected/changed remote content.

    Returns:
        Loaded tokenizer.
    """

    logger.info(
        "Loading tokenizer: %s",
        model_name,
    )

    tokenizer = AutoTokenizer.from_pretrained(
        model_name,
        revision=revision,
        trust_remote_code=True,
        use_fast=True,
    )

    if tokenizer.pad_token is None:

        if tokenizer.eos_token is None:
            raise ValueError("Tokenizer has no pad_token and no eos_token.")

        tokenizer.pad_token = tokenizer.eos_token

        logger.info("Pad token automatically set to EOS token.")

    tokenizer.padding_side = "left"

    logger.info("Tokenizer loaded successfully.")

    return tokenizer
