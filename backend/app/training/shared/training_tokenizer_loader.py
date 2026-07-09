# medical-triage-agent-ai-poc/backend/app/training/shared/training_tokenizer_loader.py

# Correctifs (audit OOM DPO, étape 2) :
#   - TOK-1 : support d'un tokenizer pré-chargé (branche Unsloth). Quand
#             runtime.engine="unsloth", FastLanguageModel.from_pretrained()
#             retourne modèle ET tokenizer couplés — il ne faut PAS
#             recharger un AutoTokenizer indépendant, sous peine
#             d'incohérence entre le tokenizer utilisé pour l'entraînement
#             et celui attendu par le modèle patché par Unsloth (source
#             probable de mismatches type "prompt vs prompt+rejected").
#   - TOK-2 : sync_with_model() ajouté — synchronise explicitement
#             model.config et model.generation_config avec le
#             pad_token_id du tokenizer, plutôt que de compter sur la
#             synchronisation automatique de TRL (qui déclenche le warning
#             "tokenizer has new PAD/BOS/EOS tokens"). Élimine le warning
#             à la source au lieu de le tolérer.

from __future__ import annotations

import logging
from typing import Any
from typing import Dict
from typing import Optional

from transformers import AutoTokenizer, PreTrainedTokenizerBase

logger = logging.getLogger(__name__)


class TrainingTokenizerLoader:
    """
    Shared tokenizer loader used by all training pipelines.

    Supported pipelines:
        - SFT
        - DPO

    Responsibilities:
        - Load tokenizer (ou réutiliser un tokenizer pré-chargé, ex. Unsloth)
        - Configure padding
        - Configure EOS token
        - Configure special tokens
        - Validate tokenizer settings
        - Synchroniser le tokenizer avec la config du modèle (pad/eos ids)

    Non-responsibilities:
        - Dataset loading
        - Model loading
        - WandB
        - Checkpoints
        - HF Hub uploads
        - Clinical evaluation
    """

    def __init__(
        self,
        config: Dict[str, Any],
        preloaded_tokenizer: Optional[PreTrainedTokenizerBase] = None,
    ) -> None:
        self.config = config
        # FIX TOK-1 — tokenizer déjà chargé (typiquement par
        # FastLanguageModel.from_pretrained() côté Unsloth). Quand fourni,
        # load_tokenizer() ne fait aucun appel réseau/disque supplémentaire
        # et se contente de le retourner tel quel.
        self._preloaded_tokenizer = preloaded_tokenizer

    def load_tokenizer(self):
        """
        Load tokenizer from Hugging Face Hub or local path, ou retourne
        le tokenizer pré-chargé fourni (branche Unsloth).
        """

        if self._preloaded_tokenizer is not None:
            logger.info(
                "Tokenizer pré-chargé fourni (runtime.engine=unsloth) — "
                "aucun AutoTokenizer.from_pretrained() supplémentaire."
            )
            return self._preloaded_tokenizer

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
            logger.info("Tokenizer has no pad token. Using EOS token.")

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

        if self._preloaded_tokenizer is not None:
            # FIX TOK-1 — un tokenizer Unsloth a déjà ses tokens spéciaux
            # correctement configurés par FastLanguageModel. Ajouter des
            # special_tokens ici referait un resize d'embeddings côté
            # modèle sans passer par les hooks Unsloth, ce qui casserait
            # la cohérence modèle/tokenizer patchée par Unsloth.
            special_tokens = self.config.get("special_tokens", {})
            if special_tokens:
                logger.warning(
                    "config['special_tokens'] est défini mais ignoré : "
                    "le tokenizer provient d'Unsloth (déjà configuré). "
                    "Retirer cette section du YAML si runtime.engine="
                    "'unsloth', ou la traiter via la config Unsloth."
                )
            return tokenizer

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
            raise ValueError("Tokenizer EOS token is not configured.")

        if tokenizer.pad_token is None:
            raise ValueError("Tokenizer PAD token is not configured.")

        logger.info("Tokenizer validation successful")

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

    # ------------------------------------------------------------------
    # FIX TOK-2 — synchronisation explicite modèle/tokenizer
    # ------------------------------------------------------------------
    @staticmethod
    def sync_with_model(model, tokenizer) -> None:
        """
        Synchronise explicitement model.config et model.generation_config
        avec le pad_token_id du tokenizer.

        Sans cet appel, TRL effectue cette synchronisation lui-même au
        premier pas d'entraînement et émet le warning "The tokenizer has
        new PAD/BOS/EOS tokens...". Le comportement final est identique,
        mais le faire explicitement ici documente l'intention et supprime
        le warning (puisque la synchronisation est déjà faite avant que
        TRL ne la déclenche).

        À appeler après le chargement du modèle ET du tokenizer, dans
        train_sft.py / train_dpo.py :

            tokenizer = TrainingTokenizerLoader.build(config=CONFIG, ...)
            model = TrainingModelLoader.build(config=CONFIG)
            TrainingTokenizerLoader.sync_with_model(model, tokenizer)
        """
        if tokenizer.pad_token_id is None:
            logger.warning(
                "tokenizer.pad_token_id est None — synchronisation "
                "modèle/tokenizer ignorée."
            )
            return

        if hasattr(model, "config"):
            model.config.pad_token_id = tokenizer.pad_token_id
            if tokenizer.eos_token_id is not None:
                model.config.eos_token_id = tokenizer.eos_token_id
            if tokenizer.bos_token_id is not None:
                model.config.bos_token_id = tokenizer.bos_token_id

        if (
            hasattr(model, "generation_config") and model.generation_config is not None
        ):  # noqa: E501
            model.generation_config.pad_token_id = tokenizer.pad_token_id
            if tokenizer.eos_token_id is not None:
                model.generation_config.eos_token_id = (
                    tokenizer.eos_token_id
                )  # noqa: E501
            if tokenizer.bos_token_id is not None:
                model.generation_config.bos_token_id = (
                    tokenizer.bos_token_id
                )  # noqa: E501

        logger.info(
            "Modèle synchronisé avec le tokenizer : pad_token_id=%s "
            "eos_token_id=%s bos_token_id=%s",
            tokenizer.pad_token_id,
            tokenizer.eos_token_id,
            tokenizer.bos_token_id,
        )

    @classmethod
    def build(
        cls,
        config: Dict[str, Any],
        preloaded_tokenizer: Optional[PreTrainedTokenizerBase] = None,
    ):
        """
        Convenience entrypoint.

        Example (transformers natif) :
            tokenizer = TrainingTokenizerLoader.build(config)

        Example (Unsloth — modèle chargé en premier) :
            model = TrainingModelLoader.build(config)
            model_loader_instance = ...  # cf. note d'intégration
            tokenizer = TrainingTokenizerLoader.build(
                config, preloaded_tokenizer=model_loader_instance.unsloth_tokenizer  # noqa: E501
            )
        """

        return cls(
            config, preloaded_tokenizer=preloaded_tokenizer
        ).prepare_for_training()  # noqa: E501
