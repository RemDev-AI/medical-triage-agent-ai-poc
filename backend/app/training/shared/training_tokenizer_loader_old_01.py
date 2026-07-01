# medical-triage-agent-ai-poc/backend/app/training/shared/training_tokenizer_loader.py

from __future__ import annotations

import logging
from typing import Any
from typing import Dict

from transformers import AutoTokenizer

logger = logging.getLogger(__name__)


class TrainingTokenizerLoader:
    """
    Shared tokenizer loader used by all training pipelines.

    Supported pipelines:
        - SFT
        - DPO

    Responsibilities:
        - Load tokenizer
        - Configure padding
        - Configure EOS token
        - Configure special tokens
        - Validate tokenizer settings

    Non-responsibilities:
        - Dataset loading
        - Model loading
        - WandB
        - Checkpoints
        - HF Hub uploads
        - Clinical evaluation
    """

    def __init__(self, config: Dict[str, Any]) -> None:
        self.config = config

    def load_tokenizer(self):
        """
        Load tokenizer from Hugging Face Hub or local path.
        """

        model_name = self.config["model"]["base_model"]

        logger.info(
            "Loading tokenizer: %s",
            model_name,
        )

        tokenizer = AutoTokenizer.from_pretrained(
            pretrained_model_name_or_path=model_name,
            trust_remote_code=self.config["model"].get(
                "trust_remote_code",
                True,
            ),
            use_fast=self.config["tokenizer"].get(
                "use_fast",
                True,
            ),
        )

        logger.info("Tokenizer loaded successfully")

        return tokenizer

    def configure_padding(
        self,
        tokenizer,
    ):
        """
        Configure padding strategy.
        """

        padding_side = self.config["tokenizer"].get(
            "padding_side",
            "right",
        )

        tokenizer.padding_side = padding_side

        if tokenizer.pad_token is None:
            logger.info(
                "Tokenizer has no pad token. Using EOS token."
            )

            tokenizer.pad_token = tokenizer.eos_token

        logger.info(
            "Padding configured: side=%s pad_token=%s",
            tokenizer.padding_side,
            tokenizer.pad_token,
        )

        return tokenizer

    def configure_special_tokens(
        self,
        tokenizer,
    ):
        """
        Configure optional special tokens.
        """

        special_tokens = self.config.get(
            "special_tokens",
            {},
        )

        tokens_to_add = {}

        if special_tokens.get("bos_token"):
            tokens_to_add["bos_token"] = special_tokens["bos_token"]

        if special_tokens.get("eos_token"):
            tokens_to_add["eos_token"] = special_tokens["eos_token"]

        if special_tokens.get("pad_token"):
            tokens_to_add["pad_token"] = special_tokens["pad_token"]

        if special_tokens.get("unk_token"):
            tokens_to_add["unk_token"] = special_tokens["unk_token"]

        if tokens_to_add:
            logger.info(
                "Adding special tokens: %s",
                list(tokens_to_add.keys()),
            )

            tokenizer.add_special_tokens(tokens_to_add)

        return tokenizer

    def configure_max_length(
        self,
        tokenizer,
    ):
        """
        Configure tokenizer maximum sequence length.
        """

        max_length = self.config["tokenizer"].get(
            "model_max_length",
            2048,
        )

        tokenizer.model_max_length = max_length

        logger.info(
            "Tokenizer max length configured: %s",
            max_length,
        )

        return tokenizer

    def validate_tokenizer(
        self,
        tokenizer,
    ) -> None:
        """
        Validate tokenizer configuration.
        """

        if tokenizer.eos_token is None:
            raise ValueError(
                "Tokenizer EOS token is not configured."
            )

        if tokenizer.pad_token is None:
            raise ValueError(
                "Tokenizer PAD token is not configured."
            )

        logger.info(
            "Tokenizer validation successful"
        )

        logger.info(
            "Vocabulary size: %s",
            len(tokenizer),
        )

        logger.info(
            "EOS token: %s",
            tokenizer.eos_token,
        )

        logger.info(
            "PAD token: %s",
            tokenizer.pad_token,
        )

    def prepare_for_training(self):
        """
        Complete tokenizer preparation pipeline.
        """

        tokenizer = self.load_tokenizer()

        tokenizer = self.configure_padding(
            tokenizer=tokenizer,
        )

        tokenizer = self.configure_special_tokens(
            tokenizer=tokenizer,
        )

        tokenizer = self.configure_max_length(
            tokenizer=tokenizer,
        )

        self.validate_tokenizer(
            tokenizer=tokenizer,
        )

        return tokenizer

    @classmethod
    def build(
        cls,
        config: Dict[str, Any],
    ):
        """
        Convenience entrypoint.

        Example:
            tokenizer = TrainingTokenizerLoader.build(config)
        """

        return cls(config).prepare_for_training()
