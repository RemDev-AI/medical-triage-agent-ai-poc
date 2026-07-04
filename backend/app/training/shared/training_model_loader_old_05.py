# medical-triage-agent-ai-poc/backend/app/training/shared/training_model_loader.py

# Corrections : bug #2 (config LoRA passée), bug #3 (torch_dtype auto),
#               bug #4 (use_reentrant=False pour Qwen3)
# Étape 1 (audit OOM DPO) : bug #5 QLoRA 4 bits (BitsAndBytesConfig)
#                           + branche Unsloth (FastLanguageModel)

from __future__ import annotations

import logging
from typing import Any, Dict, Optional  # noqa: F401

import torch
from transformers import AutoModelForCausalLM, BitsAndBytesConfig

from backend.app.training.lora.peft_setup import setup_peft_model

logger = logging.getLogger(__name__)


class TrainingModelLoader:
    """
    Shared model loader — SFT & DPO pipelines.

    Supporte deux moteurs (config["runtime"]["engine"]) :
      - "transformers" (défaut) : AutoModelForCausalLM (+ QLoRA 4 bits
        optionnel via BitsAndBytesConfig) + PEFT natif.
      - "unsloth" : FastLanguageModel, qui gère lui-même le chargement
        quantifié, le patching des kernels et l'application de LoRA.
        Dans ce mode, apply_gradient_checkpointing / apply_lora du
        pipeline "transformers" ne sont PAS utilisées.
    """

    def __init__(self, config: Dict[str, Any]) -> None:
        self.config = config
        self.engine = self.config.get("runtime", {}).get("engine", "transformers")  # noqa: E501

    # ------------------------------------------------------------------
    # Point d'entrée unique
    # ------------------------------------------------------------------
    def prepare_for_training(self) -> AutoModelForCausalLM:
        if self.engine == "unsloth":
            model = self._prepare_with_unsloth()
        else:
            model = self._prepare_with_transformers()

        self._log_trainable_parameters(model)
        return model

    # ------------------------------------------------------------------
    # Branche Transformers + PEFT natif (avec QLoRA 4 bits optionnel)
    # ------------------------------------------------------------------
    def _prepare_with_transformers(self) -> AutoModelForCausalLM:
        model = self.load_base_model()
        model = self.apply_gradient_checkpointing(model=model)
        model = self.apply_lora(model=model)
        return model

    def load_base_model(self) -> AutoModelForCausalLM:
        model_name = self.config["model"]["base_model"]

        quantization_config = self._resolve_quantization_config()

        logger.info(
            "Loading base model: %s (quantization=%s)",
            model_name,
            "4bit" if quantization_config is not None else "none",
        )

        load_kwargs: Dict[str, Any] = dict(
            pretrained_model_name_or_path=model_name,
            trust_remote_code=self.config["model"].get("trust_remote_code", True),  # noqa: E501
            device_map=self.config["model"].get("device_map", "auto"),
        )

        resolved_dtype = self._resolve_torch_dtype()

        # FIX BF16/FP16 MISMATCH — torch_dtype DOIT être transmis même
        # quand quantization_config est fourni. bnb_4bit_compute_dtype
        # ne pilote QUE le calcul des couches quantifiées en 4 bits ; les
        # couches non quantifiées (embeddings, layernorms, et donc les
        # modules ensuite enveloppés par LoRA) prennent sinon le
        # torch_dtype par défaut du config.json du checkpoint (bfloat16
        # pour Qwen3), indépendamment du GPU détecté. Sur T4 (fp16 requis,
        # pas de GradScaler compatible bf16), cela provoquait des
        # paramètres LoRA entraînables en bfloat16 alors que le Trainer
        # est configuré en fp16 → NotImplementedError sur
        # _amp_foreach_non_finite_check_and_unscale_. On force donc
        # explicitement le même dtype résolu pour les deux, quantifié ou non.  # noqa: E501
        load_kwargs["torch_dtype"] = resolved_dtype

        if quantization_config is not None:
            load_kwargs["quantization_config"] = quantization_config

        model = AutoModelForCausalLM.from_pretrained(**load_kwargs)

        logger.info("Base model loaded successfully.")
        return model

    def apply_gradient_checkpointing(
        self,
        model: AutoModelForCausalLM,
    ) -> AutoModelForCausalLM:
        enabled = self.config["training"].get("gradient_checkpointing", True)

        if enabled:
            logger.info("Enabling gradient checkpointing (use_reentrant=False).")  # noqa: E501

            # FIX BUG #4 — use_reentrant=False requis pour Qwen3
            # + DataCollatorForSeq2Seq (évite NaN silencieux sur certaines versions)  # noqa: E501
            model.gradient_checkpointing_enable(
                gradient_checkpointing_kwargs={"use_reentrant": False}
            )

            if hasattr(model, "enable_input_require_grads"):
                model.enable_input_require_grads()

        return model

    def apply_lora(
        self,
        model: AutoModelForCausalLM,
    ) -> AutoModelForCausalLM:
        logger.info("Applying LoRA adapters.")

        # FIX BUG #2 — config passée à setup_peft_model (était commentée)
        # setup_peft_model() est responsable d'appeler
        # prepare_model_for_kbit_training() si (et seulement si) le
        # modèle est effectivement quantifié (cf. peft_setup.py).
        model = setup_peft_model(
            model=model,
            config=self.config,     # ← décommenté : la config YAML est lue
        )

        logger.info("LoRA adapters applied successfully.")
        return model

    # ------------------------------------------------------------------
    # Branche Unsloth
    # ------------------------------------------------------------------
    def _prepare_with_unsloth(self) -> AutoModelForCausalLM:
        try:
            from unsloth import FastLanguageModel
        except ImportError as exc:
            raise ImportError(
                "runtime.engine='unsloth' requiert le package 'unsloth'. "
                "Installez-le (pip install unsloth) ou repassez "
                "runtime.engine='transformers' dans la config."
            ) from exc

        model_name = self.config["model"]["base_model"]
        quantization = self.config.get("quantization", {})
        load_in_4bit = quantization.get("enabled", True)
        max_seq_length = self.config["model"].get("max_seq_length", 2048)

        logger.info(
            "[unsloth] Loading base model: %s (load_in_4bit=%s)",
            model_name,
            load_in_4bit,
        )

        resolved_dtype = self._resolve_torch_dtype()  # réutilise la logique custom validée # noqa: E501

        model, tokenizer = FastLanguageModel.from_pretrained(
            model_name=model_name,
            max_seq_length=max_seq_length,
            # dtype=None,  # Unsloth choisit automatiquement bf16/fp16 selon le GPU  # noqa: E501
            dtype=resolved_dtype,  # source unique de vérité, plus de dtype=None # noqa: E501
            load_in_4bit=load_in_4bit,
        )

        # Le tokenizer chargé par Unsloth est stocké pour être récupéré
        # par training_tokenizer_loader.py si besoin (cohérence des
        # tokens spéciaux avec le modèle patché par Unsloth).
        self.unsloth_tokenizer = tokenizer

        lora_cfg = self.config.get("lora", {})

        logger.info("[unsloth] Applying LoRA adapters via get_peft_model.")

        model = FastLanguageModel.get_peft_model(
            model,
            r=lora_cfg.get("r", 16),
            target_modules=lora_cfg.get(
                "target_modules",
                ["q_proj", "k_proj", "v_proj", "o_proj",
                 "gate_proj", "up_proj", "down_proj"],
            ),
            lora_alpha=lora_cfg.get("lora_alpha", 16),
            lora_dropout=lora_cfg.get("lora_dropout", 0.0),
            bias=lora_cfg.get("bias", "none"),
            # FIX BUG #4 (équivalent Unsloth) — "unsloth" active un
            # gradient checkpointing optimisé (kernels custom),
            # cohérent avec use_reentrant=False côté Transformers.
            use_gradient_checkpointing=(
                "unsloth"
                if self.config["training"].get("gradient_checkpointing", True)
                else False
            ),
            random_state=self.config.get("seed", 42),
        )

        # --- Garde-fou fp16 (même correctif que peft_setup.py, étape 09) ---
        # Unsloth peut caster les adapters LoRA en bfloat16 en interne sur
        # T4, indépendamment de dtype=resolved_dtype passé à
        # from_pretrained() (cf. audit dtype : 393 paramètres entraînables
        # tous en bfloat16 malgré resolved_dtype=float16). On force donc
        # explicitement tout paramètre entraînable resté en bfloat16 vers
        # float16, seul dtype compatible avec le GradScaler sur T4.
        n_cast = 0
        for param in model.parameters():
            if param.requires_grad and param.dtype == torch.bfloat16:
                param.data = param.data.to(torch.float16)
                n_cast += 1

        if n_cast:
            logger.warning(
                "[unsloth] Garde-fou fp16 : %d paramètre(s) entraînable(s) "
                "recastés de bfloat16 vers float16.",
                n_cast,
            )

        logger.info("[unsloth] Base model + LoRA ready.")
        return model

    # ------------------------------------------------------------------
    # Résolution quantification / dtype
    # ------------------------------------------------------------------
    def _resolve_quantization_config(self) -> Optional[BitsAndBytesConfig]:
        """
        Construit un BitsAndBytesConfig si config["quantization"]["enabled"]
        est vrai. Reste rétro-compatible : si la section "quantization"
        est absente de la config, aucune quantification n'est appliquée
        (comportement historique — FP16 complet).
        """
        quant_cfg = self.config.get("quantization", {})

        if not quant_cfg.get("enabled", False):
            return None

        compute_dtype = self._resolve_torch_dtype()

        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type=quant_cfg.get("bnb_4bit_quant_type", "nf4"),
            bnb_4bit_compute_dtype=compute_dtype,
            bnb_4bit_use_double_quant=quant_cfg.get(
                "bnb_4bit_use_double_quant", True
            ),
        )

        logger.info(
            "QLoRA 4 bits activé : quant_type=%s compute_dtype=%s double_quant=%s",  # noqa: E501
            bnb_config.bnb_4bit_quant_type,
            compute_dtype,
            bnb_config.bnb_4bit_use_double_quant,
        )

        return bnb_config

    # FIX BUG #3 — torch_dtype "auto" délégué à colab_environment
    def _resolve_torch_dtype(self) -> torch.dtype:
        dtype = self.config["model"].get("torch_dtype", "auto")

        if dtype == "auto":
            # Source unique de vérité : même logique que apply_precision_arguments()  # noqa: E501
            from backend.app.training.colab.colab_environment import (
                get_training_dtype,
            )
            resolved = get_training_dtype()
            logger.info("torch_dtype=auto → résolu par runtime : %s", resolved)
            return resolved

        mapping = {
            "float16": torch.float16,
            "fp16": torch.float16,
            "bfloat16": torch.bfloat16,
            "bf16": torch.bfloat16,
            "float32": torch.float32,
            "fp32": torch.float32,
        }
        resolved = mapping.get(dtype.lower(), torch.float16)
        logger.info("torch_dtype=%s → %s", dtype, resolved)
        return resolved

    @staticmethod
    def _log_trainable_parameters(model: AutoModelForCausalLM) -> None:
        trainable_params = 0
        total_params = 0

        for parameter in model.parameters():
            total_params += parameter.numel()
            if parameter.requires_grad:
                trainable_params += parameter.numel()

        percentage = (
            100 * trainable_params / total_params if total_params > 0 else 0.0
        )

        logger.info(
            "Trainable parameters: %s / %s (%.4f%%)",
            f"{trainable_params:,}",
            f"{total_params:,}",
            percentage,
        )

    @classmethod
    def build(cls, config: Dict[str, Any]) -> AutoModelForCausalLM:
        return cls(config).prepare_for_training()
