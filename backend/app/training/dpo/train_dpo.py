# medical-triage-agent-ai-poc/backend/app/training/dpo/train_dpo.py
#
# Corrections appliquées (alignées sur les fixes SFT) :
#   - Bug #2  : config LoRA passée via TrainingModelLoader (déjà corrigé côté loader)  # noqa : E501
#   - Bug #3  : fp16/bf16 supprimés du YAML, torch_dtype="auto", source unique
#   - Bug #4  : use_reentrant=False (déjà corrigé côté loader)
#   - Bug #5  : NaNGuardCallback ajouté
#   - DPO-1   : EarlyStoppingCallback absent du build_trainer → ajouté
#   - DPO-2   : DPOConfig ne supporte pas tous les champs TrainingArguments
#               → champs manquants ajoutés (eval_strategy, save_strategy,
#                 load_best_model_at_end, warmup_ratio, lr_scheduler_type,
#                 max_grad_norm, dataloader_num_workers)
#   - DPO-3   : max_length lu depuis la config YAML
#   - DPO-4   : sous-ensemble de validation (max_train_samples / max_val_samples)  # noqa : E501
#   - DPO-5   : SafetyFilter appliqué sur chosen/rejected avant entraînement
#   - DPO-6   : reference_free (YAML) était défini mais jamais transmis à
#               DPOConfig → ajouté à build_dpo_config().
#   - GC-1    : gradient_checkpointing_kwargs={"use_reentrant": False} ajouté
#               à DPOConfig (même correctif que train_sft.py). Sans lui,
#               DPOTrainer réactive le gradient checkpointing à sa façon au
#               démarrage de trainer.train(), écrasant le use_reentrant=False
#               déjà posé par TrainingModelLoader (bug #4, requis Qwen3).
#   - TOK-1/TOK-2 : modèle chargé AVANT le tokenizer (nécessaire pour la
#               branche Unsloth, qui charge les deux ensemble), et
#               synchronisation explicite model/tokenizer (pad/eos/bos ids)
#               pour éliminer le warning PAD/BOS/EOS à la source.
#
# CORRECTIONS OOM :
#   - OOM-1   : max_length 2048 → 512 dans le YAML (activations ∝ seq²)
#   - OOM-2   : max_prompt_length 1024 → 256 dans le YAML (config uniquement)
#   - OOM-3   : fp16/bf16 supprimés du YAML, torch_dtype="auto" uniquement
#   - OOM-4   : max_prompt_length SUPPRIMÉ de DPOConfig — absent de cette version TRL  # noqa : E501
#               max_length seul est transmis à DPOConfig
#   - OOM-5   : PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True défini au démarrage  # noqa : E501
#   - OOM-6   : gestion explicite du ref_model (audit hypothèse #1).
#               Le projet utilise EXCLUSIVEMENT LoRA (SFT et DPO) — aucun
#               scénario de full fine-tuning n'est prévu dans la mission.
#               - Si `model` est un PeftModel (cas nominal, toujours vrai en
#                 pratique) : ref_model=None. DPOTrainer désactive les
#                 adapters sur le modèle courant pour calculer πref — AUCUNE
#                 copie mémoire supplémentaire. C'est le comportement le
#                 plus économe.
#               - Si `model` n'est PAS un PeftModel (régression de config,
#                 ex. LoRA désactivé par erreur) : on lève une erreur
#                 explicite plutôt que de charger un second modèle complet.
#                 DPOTrainer sans ref_model explicite deep-copierait `model`
#                 en interne (double la VRAM, cause de l'OOM identifiée par
#                 l'audit) — ce chemin n'est pas testé en prod et ne doit
#                 pas être emprunté silencieusement.

from __future__ import annotations

import json
import logging
import math
import os
import torch
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import yaml
from datasets import Dataset, load_dataset

try:
    from trl import DPOConfig, DPOTrainer
except Exception:
    DPOConfig = None
    DPOTrainer = None

from peft import PeftModel
from transformers import EarlyStoppingCallback, TrainerCallback

from backend.app.training.colab.colab_environment import (
    apply_precision_arguments,
    resolve_quantization_settings,
)
from backend.app.training.shared.training_model_loader import (
    TrainingModelLoader,
)
from backend.app.training.shared.training_tokenizer_loader import (
    TrainingTokenizerLoader,
)
from backend.app.training.shared.training_utils import (
    TrainingUtils,
)
from backend.app.training.colab.colab_checkpoint_sync import (
    create_default_checkpoint_sync,
)

logger = logging.getLogger(__name__)

CONFIG_PATH = Path(__file__).parent / "dpo_config_validation.yaml"


def load_config() -> Dict:
    with open(CONFIG_PATH, "r", encoding="utf-8") as file:
        return yaml.safe_load(file)


CONFIG = load_config()
# FIX QUANT-2 — complète config["quantization"] selon le GPU détecté si
# absent du YAML ; conserve tel quel un choix explicite déjà présent
# (avec warning informatif en cas de désaccord). cf. colab_environment.py.
CONFIG = resolve_quantization_settings(CONFIG)


# ---------------------------------------------------------------------------
# OOM-5 — Définir PYTORCH_CUDA_ALLOC_CONF avant tout import torch
# Réduit la fragmentation mémoire sur T4 / Colab.
# Doit être défini avant que PyTorch alloue quoi que ce soit.
# ---------------------------------------------------------------------------
def _configure_cuda_allocator() -> None:
    current = os.environ.get("PYTORCH_CUDA_ALLOC_CONF", "")
    if "expandable_segments" not in current:
        os.environ["PYTORCH_CUDA_ALLOC_CONF"] = (
            (current + ",expandable_segments:True").lstrip(",")
        )
        logger.info(
            "PYTORCH_CUDA_ALLOC_CONF → %s",
            os.environ["PYTORCH_CUDA_ALLOC_CONF"],
        )


_configure_cuda_allocator()


# ---------------------------------------------------------------------------
# FIX BUG #5 — NaNGuardCallback (aligné sur SFT)
# ---------------------------------------------------------------------------
class NaNGuardCallback(TrainerCallback):
    def on_evaluate(self, args, state, control, metrics, **kwargs):
        loss = metrics.get("eval_loss", 0.0)
        if math.isnan(loss) or math.isinf(loss):
            raise ValueError(
                f"eval_loss={loss} détecté à step {state.global_step}. "
                "Arrêt du run DPO. Vérifier les données chosen/rejected."
            )


# ---------------------------------------------------------------------------
# FIX DPO-5 — SafetyFilter
# Écarte les exemples dont chosen ou rejected contiennent
# des keywords dangereux définis dans dpo_config.yaml [safety]
# ---------------------------------------------------------------------------
class SafetyFilter:
    def __init__(self, config: Dict) -> None:
        safety = config.get("safety", {})
        self.blocked: List[str] = (
            safety.get("hallucination_keywords", [])
            + safety.get("dangerous_keywords", [])
        )

    def is_safe(self, example: Dict) -> bool:
        for field in ("chosen", "rejected", "prompt"):
            text = example.get(field, "").lower()
            for kw in self.blocked:
                if kw.lower() in text:
                    logger.warning(
                        "Exemple écarté — keyword dangereux '%s' "
                        "dans le champ '%s'.",
                        kw,
                        field,
                    )
                    return False
        return True


def load_jsonl_dataset(path: str) -> Dataset:
    records: List[Dict] = []
    with open(path, "r", encoding="utf-8") as file:
        for line in file:
            records.append(json.loads(line))
    return Dataset.from_list(records)


def load_hf_dataset(
    dataset_repo: str,
    dataset_config: str,
    split: str,
) -> Dataset:
    return load_dataset(
        path=dataset_repo,
        name=dataset_config,
        split=split,
    )


def load_dataset_source(split: str) -> Dataset:
    dataset_config = CONFIG["dataset"]

    hf_repo = dataset_config.get("hf_repo")
    hf_config = dataset_config.get("hf_config", "dpo")

    if hf_repo:
        dataset = load_hf_dataset(
            dataset_repo=hf_repo,
            dataset_config=hf_config,
            split=split,
        )
    else:
        path_mapping = {
            "train": "train_path",
            "validation": "validation_path",
            "test": "test_path",
            "clinical_eval": "clinical_eval_path",
        }
        dataset = load_jsonl_dataset(dataset_config[path_mapping[split]])

    # FIX DPO-4 — sous-ensemble pour run de validation rapide
    max_key = "max_train_samples" if split == "train" else "max_val_samples"
    max_samples = dataset_config.get(max_key)
    if max_samples:
        dataset = dataset.select(range(min(max_samples, len(dataset))))
        logger.info("Split '%s' limité à %d exemples.", split, len(dataset))

    return dataset


def build_dpo_sample(example: Dict) -> Dict:
    return {
        "prompt": example["prompt"],
        "chosen": example["chosen"],
        "rejected": example["rejected"],
    }


def prepare_datasets() -> Tuple[Dataset, Dataset]:
    safety_filter = SafetyFilter(CONFIG)

    train_dataset = load_dataset_source("train").map(build_dpo_sample)
    validation_dataset = load_dataset_source("validation").map(build_dpo_sample)  # noqa: E501

    # FIX DPO-5 — filtre de sécurité sur les données DPO
    before_train = len(train_dataset)
    before_val = len(validation_dataset)

    train_dataset = train_dataset.filter(safety_filter.is_safe)
    validation_dataset = validation_dataset.filter(safety_filter.is_safe)

    logger.info(
        "SafetyFilter — train : %d → %d exemples (-%d écartés).",
        before_train, len(train_dataset), before_train - len(train_dataset),
    )
    logger.info(
        "SafetyFilter — validation : %d → %d exemples (-%d écartés).",
        before_val, len(validation_dataset), before_val - len(validation_dataset),  # noqa: E501
    )

    if len(validation_dataset) == 0:
        raise ValueError(
            "Validation dataset vide après SafetyFilter. "
            "Vérifier les keywords dans dpo_config.yaml [safety]."
        )

    return train_dataset, validation_dataset


def build_dpo_config() -> "DPOConfig":
    training_config = CONFIG["training"]
    dpo_config = CONFIG["dpo"]

    max_length = dpo_config.get("max_length", 512)
    # max_prompt_length conservé localement pour la validation logique
    # uniquement — NE DOIT PAS être transmis à DPOConfig (absent de TRL ici).
    max_prompt_length = dpo_config.get("max_prompt_length", 256)

    # Validation : max_prompt_length doit être < max_length
    if max_prompt_length >= max_length:
        raise ValueError(
            f"max_prompt_length ({max_prompt_length}) doit être "
            f"strictement inférieur à max_length ({max_length})."
        )

    reference_free = bool(dpo_config.get("reference_free", False))

    training_args = {
        "output_dir": training_config["output_dir"],
        "num_train_epochs": training_config["num_train_epochs"],
        "per_device_train_batch_size": training_config["per_device_train_batch_size"],  # noqa: E501
        "per_device_eval_batch_size": training_config["per_device_eval_batch_size"],   # noqa: E501
        "gradient_accumulation_steps": training_config["gradient_accumulation_steps"],  # noqa: E501
        "learning_rate": float(training_config["learning_rate"]),
        "logging_steps": training_config["logging_steps"],
        "eval_steps": training_config["eval_steps"],
        "save_steps": training_config["save_steps"],
        "save_total_limit": training_config["save_total_limit"],
        "gradient_checkpointing": training_config["gradient_checkpointing"],
        # FIX GC-1 (aligné sur train_sft.py) — le Trainer/DPOTrainer HF
        # réactive lui-même le gradient checkpointing au démarrage de
        # trainer.train() (puisque gradient_checkpointing=True ci-dessus),
        # en écrasant potentiellement le use_reentrant=False déjà posé par
        # TrainingModelLoader.apply_gradient_checkpointing() (bug #4,
        # requis pour Qwen3). Rendu explicite ici pour rester cohérent
        # quelle que soit la source de l'activation.
        "gradient_checkpointing_kwargs": {"use_reentrant": False},
        "report_to": training_config.get("report_to", ["wandb"]),
        "dataloader_num_workers": training_config.get("dataloader_num_workers", 0),  # noqa: E501

        # FIX DPO-2 — champs TrainingArguments
        "eval_strategy": training_config.get("evaluation_strategy", "steps"),
        "save_strategy": training_config.get("save_strategy", "steps"),
        "load_best_model_at_end": training_config.get("load_best_model_at_end", True),   # noqa: E501
        "metric_for_best_model": training_config.get("metric_for_best_model", "eval_loss"),  # noqa: E501
        "greater_is_better": training_config.get("greater_is_better", False),
        "warmup_ratio": float(training_config.get("warmup_ratio", 0.05)),
        "lr_scheduler_type": training_config.get("lr_scheduler_type", "cosine"),  # noqa: E501
        "max_grad_norm": float(training_config.get("max_grad_norm", 1.0)),

        # OOM-4 : max_prompt_length SUPPRIMÉ — absent de DPOConfig dans
        # cette version de TRL. max_length seul est transmis.
        "beta": float(dpo_config.get("beta", 0.1)),
        "max_length": max_length,
        "loss_type": dpo_config.get("loss_type", "sigmoid"),
        "truncation_mode": dpo_config.get("truncation_mode", "keep_end"),

        # FIX DPO-6 — reference_free était lu du YAML mais jamais transmis.
        "reference_free": reference_free,
    }

    # FIX BUG #3 — précision détectée depuis le GPU (source unique)
    # OOM-3 : fp16/bf16 NE DOIVENT PAS être dans le YAML,
    # apply_precision_arguments() est la seule source de vérité.
    training_args = apply_precision_arguments(training_args)

    return DPOConfig(**training_args)


# ---------------------------------------------------------------------------
# FIX OOM-6 — Gestion explicite et sûre du ref_model
# ---------------------------------------------------------------------------
def resolve_ref_model(model) -> Optional[object]:
    """
    Détermine le ref_model à transmettre à DPOTrainer.

    Le projet (mission) n'utilise QUE LoRA pour SFT et DPO — le full
    fine-tuning n'est pas un scénario prévu. En conséquence :

    - reference_free=True (config) : aucun modèle de référence requis,
      quel que soit le type de `model`.
    - `model` est un PeftModel (cas nominal, toujours vrai en pratique
      avec ce pipeline) : retourne None. DPOTrainer désactivera les
      adapters LoRA sur `model` lui-même pour calculer πref — pas de
      copie mémoire supplémentaire.
    - `model` n'est PAS un PeftModel : ceci indique une régression de
      config (LoRA désactivé par erreur, ou pipeline détourné de son
      usage prévu). On lève une erreur explicite plutôt que de charger
      un second modèle complet — ce chemin n'est ni prévu par la
      mission ni testé, et laisser DPOTrainer deep-copier `model` en
      interne reproduirait silencieusement l'OutOfMemoryError identifié
      par l'audit.
    """
    dpo_config = CONFIG.get("dpo", {})

    if dpo_config.get("reference_free", False):
        logger.info(
            "reference_free=True — aucun ref_model chargé (πref non requis)."
        )
        return None

    if isinstance(model, PeftModel):
        logger.info(
            "Policy model est un PeftModel (LoRA/QLoRA actif) — "
            "ref_model=None. DPOTrainer désactivera les adapters pour "
            "calculer πref sur le même modèle (aucune copie mémoire "
            "supplémentaire)."
        )
        return None

    raise RuntimeError(
        "DPO training attend un policy model chargé avec LoRA "
        "(PeftModel) — ce pipeline ne supporte pas le full fine-tuning. "
        "Le modèle reçu n'est pas un PeftModel : vérifier que "
        "config['lora'] est bien défini et que TrainingModelLoader.build() "
        "applique correctement les adapters LoRA. Sans cela, DPOTrainer "
        "deep-copierait le modèle complet en interne et reproduirait "
        "l'OutOfMemoryError identifié dans l'audit. Si le full "
        "fine-tuning devient un besoin réel, il faudra alors charger "
        "explicitement un ref_model séparé (quantifié à l'identique) "
        "plutôt que de contourner cette vérification."
    )


def build_trainer(
    model,
    tokenizer,
    train_dataset,
    validation_dataset,
) -> "DPOTrainer":
    if DPOTrainer is None:
        raise ImportError(
            "TRL package is required for DPO training. "
            "pip install trl --break-system-packages"
        )

    ref_model = resolve_ref_model(model)

    # FIX DPO-1 — EarlyStoppingCallback absent dans la version originale
    early_stopping = EarlyStoppingCallback(
        early_stopping_patience=CONFIG["training"].get("early_stopping_patience", 2)  # noqa: E501
    )

    # NaNGuardCallback en premier pour stopper avant EarlyStopping
    nan_guard = NaNGuardCallback()

    return DPOTrainer(
        model=model,
        ref_model=ref_model,
        args=build_dpo_config(),
        processing_class=tokenizer,
        train_dataset=train_dataset,
        eval_dataset=validation_dataset,
        callbacks=[nan_guard, early_stopping],
    )


def publish_training_artifacts() -> None:
    output_dir = Path(CONFIG["training"]["output_dir"])
    metadata = TrainingUtils.build_training_metadata(config=CONFIG)
    TrainingUtils.save_training_metadata(
        metadata=metadata,
        output_directory=output_dir,
    )
    logger.info("DPO training metadata saved.")


def run_clinical_evaluation(model, tokenizer) -> None:
    logger.info("Clinical DPO evaluation integration pending.")


def train(publish_to_hf: bool = False):   # False par défaut en validation
    TrainingUtils.setup_logging(CONFIG["system"]["logging_level"])
    logger.info("Starting DPO validation run.")

    TrainingUtils.set_seed(CONFIG["system"]["seed"])

    wandb_run = TrainingUtils.initialize_wandb(config=CONFIG)

    # FIX TOK-1 — le modèle est chargé EN PREMIER : en mode
    # runtime.engine="unsloth", TrainingModelLoader charge modèle ET
    # tokenizer ensemble (FastLanguageModel.from_pretrained). Le tokenizer
    # pré-chargé est ensuite réutilisé par TrainingTokenizerLoader au lieu
    # d'en recharger un indépendant (source d'incohérence sinon, cf.
    # warning TRL "prompt mismatch").
    model_loader = TrainingModelLoader(CONFIG)
    model = model_loader.prepare_for_training()

    tokenizer = TrainingTokenizerLoader.build(
        config=CONFIG,
        preloaded_tokenizer=getattr(model_loader, "unsloth_tokenizer", None),
    )

    # FIX TOK-2 — synchronise explicitement model.config /
    # model.generation_config avec le tokenizer (élimine le warning
    # "tokenizer has new PAD/BOS/EOS tokens" à la source).
    TrainingTokenizerLoader.sync_with_model(model=model, tokenizer=tokenizer)

    train_dataset, validation_dataset = prepare_datasets()

    logger.info(
        "Validation run — train=%d exemples, val=%d exemples.",
        len(train_dataset), len(validation_dataset),
    )

    print("=" * 60)
    print(torch.cuda.memory_summary())
    print("=" * 60)

    trainer = build_trainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=train_dataset,
        validation_dataset=validation_dataset,
    )

    print("=" * 60)
    print(torch.cuda.memory_summary())
    print("=" * 60)
    
    checkpoint_sync = create_default_checkpoint_sync(
        output_dir=CONFIG["training"]["output_dir"],
        training_type="sft",
    )

    resume_checkpoint = (
        CONFIG["training"].get("resume_from_checkpoint")
    )

    if resume_checkpoint is None:
        resume_checkpoint = (
            checkpoint_sync.restore_latest_checkpoint_from_huggingface()
        )

    trainer.train(
        # resume_from_checkpoint=CONFIG["training"].get("resume_from_checkpoint")
        resume_from_checkpoint=resume_checkpoint
    )

    trainer.save_model()
    tokenizer.save_pretrained(CONFIG["training"]["output_dir"])
    publish_training_artifacts()
    run_clinical_evaluation(model=model, tokenizer=tokenizer)
    TrainingUtils.finalize_wandb_run(wandb_run)

    logger.info("DPO validation run completed.")


if __name__ == "__main__":
    train()
