# medical-triage-agent-ai-poc/backend/app/llm/loaders/tokenizer_loader.py

"""
Tokenizer loader for Qwen models.
"""

from __future__ import annotations

from transformers import AutoTokenizer


def load_tokenizer(
    model_name: str,
):
    """
    Load tokenizer from Hugging Face.

    Args:
        model_name:
            Hugging Face model identifier.

    Returns:
        Loaded tokenizer.
    """

    tokenizer = AutoTokenizer.from_pretrained(
        model_name,
        trust_remote_code=True,
    )

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    tokenizer.padding_side = "left"

    return tokenizer
