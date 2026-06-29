# medical-triage-agent-ai-poc/backend/app/training/dpo/train_dpo.py

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
#   - DPO-3   : max_prompt_length et max_length lus depuis la config YAML
#   - DPO-4   : sous-ensemble de validation (max_train_samples / max_val_samples)  # noqa : E501
#   - DPO-5   : SafetyFilter appliqué sur chosen/rejected avant entraînement
#
# CORRECTIONS OOM :
#   - OOM-1   : max_length 2048 → 512 dans le YAML (activations ∝ seq²)
#   - OOM-2   : max_prompt_length 1024 → 256 dans le YAML
#   - OOM-3   : fp16/bf16 supprimés du YAML, torch_dtype="auto" uniquement
#   - OOM-4   : max_prompt_length transmis à DPOConfig (était absent de build_dpo_config)  # noqa : E501
#   - OOM-5   : PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True défini au démarrage  # noqa : E501

from __future__ import annotations

import json
import logging
import math
import os
from pathlib import Path
from typing import Dict, List, Tuple

import yaml
from datasets import Dataset, load_dataset

try:
    from trl import DPOConfig, DPOTrainer
except Exception:
    DPOConfig = None
    DPOTrainer = None

from transformers import EarlyStoppingCallback, TrainerCallback

from backend.app.training.colab.colab_environment import (
    apply_precision_arguments,
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

logger = logging.getLogger(__name__)

CONFIG_PATH = Path(__file__).parent / "dpo_config_validation.yaml"


def load_config() -> Dict:
    with open(CONFIG_PATH, "r", encoding="utf-8") as file:
        return yaml.safe_load(file)


CONFIG = load_config()


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

    # OOM-4 : max_prompt_length était absent → transmis maintenant
    max_length = dpo_config.get("max_length", 512)
    max_prompt_length = dpo_config.get("max_prompt_length", 256)

    # Validation : max_prompt_length doit être < max_length
    if max_prompt_length >= max_length:
        raise ValueError(
            f"max_prompt_length ({max_prompt_length}) doit être "
            f"strictement inférieur à max_length ({max_length})."
        )

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

        # FIX DPO-3 + OOM-4 : max_length et max_prompt_length transmis
        "beta": float(dpo_config.get("beta", 0.1)),
        "max_length": max_length,
        "max_prompt_length": max_prompt_length,
        "loss_type": dpo_config.get("loss_type", "sigmoid"),
        "truncation_mode": dpo_config.get("truncation_mode", "keep_end"),
    }

    # FIX BUG #3 — précision détectée depuis le GPU (source unique)
    # OOM-3 : fp16/bf16 NE DOIVENT PAS être dans le YAML,
    # apply_precision_arguments() est la seule source de vérité.
    training_args = apply_precision_arguments(training_args)

    return DPOConfig(**training_args)


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

    # FIX DPO-1 — EarlyStoppingCallback absent dans la version originale
    early_stopping = EarlyStoppingCallback(
        early_stopping_patience=CONFIG["training"].get("early_stopping_patience", 2)  # noqa: E501
    )

    # NaNGuardCallback en premier pour stopper avant EarlyStopping
    nan_guard = NaNGuardCallback()

    return DPOTrainer(
        model=model,
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

    tokenizer = TrainingTokenizerLoader.build(config=CONFIG)
    model = TrainingModelLoader.build(config=CONFIG)

    train_dataset, validation_dataset = prepare_datasets()

    logger.info(
        "Validation run — train=%d exemples, val=%d exemples.",
        len(train_dataset), len(validation_dataset),
    )

    trainer = build_trainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=train_dataset,
        validation_dataset=validation_dataset,
    )

    trainer.train(
        resume_from_checkpoint=CONFIG["training"].get("resume_from_checkpoint")
    )

    trainer.save_model()
    tokenizer.save_pretrained(CONFIG["training"]["output_dir"])
    publish_training_artifacts()
    run_clinical_evaluation(model=model, tokenizer=tokenizer)
    TrainingUtils.finalize_wandb_run(wandb_run)

    logger.info("DPO validation run completed.")


if __name__ == "__main__":
    train()
