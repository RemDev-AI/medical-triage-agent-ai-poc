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


def load_tokenizer(
    model_name: str,
) -> PreTrainedTokenizerBase:
    """
    Load tokenizer from Hugging Face.

    Args:
        model_name:
            Hugging Face model identifier.

    Returns:
        Loaded tokenizer.
    """

    logger.info(
        "Loading tokenizer: %s",
        model_name,
    )

    tokenizer = AutoTokenizer.from_pretrained(
        model_name,
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
