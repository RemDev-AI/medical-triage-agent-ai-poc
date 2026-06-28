# medical-triage-agent-ai-poc/backend/app/training/sft/train_sft.py
# Version : validation LoRA — bugs #1 #2 #3 corrigés

from __future__ import annotations

import json
import logging
import math
from pathlib import Path
from typing import Dict, List, Tuple

import yaml
from datasets import Dataset, load_dataset
from transformers import (
    DataCollatorForSeq2Seq,
    EarlyStoppingCallback,
    Trainer,
    TrainerCallback,
    TrainingArguments,
)

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

CONFIG_PATH = Path(__file__).parent / "sft_config_validation.yaml"


def load_config() -> Dict:
    with open(CONFIG_PATH, "r", encoding="utf-8") as file:
        return yaml.safe_load(file)


CONFIG = load_config()


# ---------------------------------------------------------------------------
# FIX BUG #5 — NaNGuardCallback
# Arrêt immédiat si eval_loss = NaN pour éviter un run silencieusement cassé
# ---------------------------------------------------------------------------
class NaNGuardCallback(TrainerCallback):
    def on_evaluate(self, args, state, control, metrics, **kwargs):
        loss = metrics.get("eval_loss", 0.0)
        if math.isnan(loss) or math.isinf(loss):
            raise ValueError(
                f"eval_loss={loss} détecté à step {state.global_step}. "
                "Arrêt du run. Vérifier le masquage des labels (bug #1)."
            )


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
    hf_config = dataset_config.get("hf_config", "sft")

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

    # Sous-ensemble pour run de validation rapide
    max_key = "max_train_samples" if split == "train" else "max_val_samples"
    max_samples = dataset_config.get(max_key)
    if max_samples:
        dataset = dataset.select(range(min(max_samples, len(dataset))))
        logger.info("Split '%s' limité à %d exemples.", split, len(dataset))

    return dataset


def build_prompt(example: Dict) -> Dict:
    prompt = (
        "<|system|>\n"
        "You are a medical triage assistant.\n"
        "Provide clinically safe recommendations.\n"
        "<|user|>\n"
        f"{example['instruction']}\n"
        "<|assistant|>\n"
        f"{example['response']}"
    )
    return {"text": prompt}


# ---------------------------------------------------------------------------
# FIX BUG #1 — tokenize_function
# Vérifie qu'au moins un token est actif après masquage.
# Retourne None pour les exemples entièrement masqués (filtrés ensuite).
# ---------------------------------------------------------------------------
def tokenize_function(examples, tokenizer, max_length: int):
    outputs = tokenizer(
        examples["text"],
        truncation=True,
        padding=False,
        max_length=max_length,
    )

    labels = []
    skipped = 0

    for input_ids, text in zip(outputs["input_ids"], examples["text"]):
        assistant_marker = "<|assistant|>\n"
        parts = text.split(assistant_marker)

        if len(parts) < 2:
            logger.warning("Marqueur <|assistant|> absent — exemple masqué.")
            labels.append([-100] * len(input_ids))
            skipped += 1
            continue

        prefix = parts[0] + assistant_marker
        prefix_ids = tokenizer(
            prefix,
            truncation=True,
            max_length=max_length,
            add_special_tokens=False,
        )["input_ids"]

        prefix_len = min(len(prefix_ids), len(input_ids))
        label = [-100] * prefix_len + input_ids[prefix_len:]

        # FIX : vérification des tokens actifs
        active_tokens = sum(1 for t in label if t != -100)
        if active_tokens == 0:
            logger.warning(
                "Aucun token actif après masquage (prefix_len=%d, "
                "input_ids_len=%d) — exemple écarté.",
                prefix_len,
                len(input_ids),
            )
            # On conserve la séquence pour le filtre suivant
            labels.append([-100] * len(input_ids))
            skipped += 1
            continue

        labels.append(label)

    if skipped:
        logger.warning(
            "%d exemple(s) entièrement masqués dans ce batch — "
            "ils seront filtrés.",
            skipped,
        )

    outputs["labels"] = labels
    return outputs


def prepare_datasets(tokenizer) -> Tuple[Dataset, Dataset]:
    dataset_config = CONFIG["dataset"]

    train_dataset = load_dataset_source("train")
    validation_dataset = load_dataset_source("validation")

    train_dataset = train_dataset.map(
        build_prompt,
        remove_columns=train_dataset.column_names,
    )
    validation_dataset = validation_dataset.map(
        build_prompt,
        remove_columns=validation_dataset.column_names,
    )

    train_dataset = train_dataset.map(
        lambda x: tokenize_function(
            x, tokenizer, dataset_config["max_sequence_length"]
        ),
        batched=True,
    )
    validation_dataset = validation_dataset.map(
        lambda x: tokenize_function(
            x, tokenizer, dataset_config["max_sequence_length"]
        ),
        batched=True,
    )

    # FIX BUG #1 — filtre les exemples sans token actif
    before_train = len(train_dataset)
    before_val = len(validation_dataset)

    train_dataset = train_dataset.filter(
        lambda x: any(t != -100 for t in x["labels"])
    )
    validation_dataset = validation_dataset.filter(
        lambda x: any(t != -100 for t in x["labels"])
    )

    logger.info(
        "Train : %d → %d exemples après filtre (-%d masqués).",
        before_train, len(train_dataset), before_train - len(train_dataset),
    )
    logger.info(
        "Validation : %d → %d exemples après filtre (-%d masqués).",
        before_val, len(validation_dataset), before_val - len(validation_dataset),  # noqa : E501
    )

    if len(validation_dataset) == 0:
        raise ValueError(
            "Validation dataset vide après filtre. "
            "Tous les labels étaient masqués — vérifier build_prompt() "
            "et le marqueur <|assistant|>."
        )

    return train_dataset, validation_dataset


def build_training_arguments() -> TrainingArguments:
    training_config = CONFIG["training"]

    training_args = {
        "output_dir": training_config["output_dir"],
        "num_train_epochs": training_config["num_train_epochs"],
        "per_device_train_batch_size": training_config["per_device_train_batch_size"],  # noqa : E501
        "per_device_eval_batch_size": training_config["per_device_eval_batch_size"],  # noqa : E501
        "gradient_accumulation_steps": training_config["gradient_accumulation_steps"],  # noqa : E501
        "learning_rate": float(training_config["learning_rate"]),
        "weight_decay": float(training_config["weight_decay"]),
        "warmup_ratio": float(training_config["warmup_ratio"]),
        "logging_steps": training_config["logging_steps"],
        "eval_steps": training_config["eval_steps"],
        "save_steps": training_config["save_steps"],
        "save_total_limit": training_config["save_total_limit"],
        "eval_strategy": training_config["evaluation_strategy"],
        "save_strategy": training_config["save_strategy"],
        "load_best_model_at_end": training_config["load_best_model_at_end"],
        "metric_for_best_model": training_config["metric_for_best_model"],
        "greater_is_better": training_config["greater_is_better"],
        "gradient_checkpointing": training_config["gradient_checkpointing"],
        "lr_scheduler_type": training_config["lr_scheduler_type"],
        "max_grad_norm": float(training_config["max_grad_norm"]),
        "report_to": training_config.get("report_to", ["wandb"]),
        "dataloader_num_workers": training_config.get("dataloader_num_workers", 0),  # noqa : E501
        "remove_unused_columns": True,
    }

    # FIX BUG #3 — source unique de précision : le runtime détecte le GPU
    training_args = apply_precision_arguments(training_args)

    return TrainingArguments(**training_args)


def build_trainer(model, tokenizer, train_dataset, validation_dataset):
    training_args = build_training_arguments()

    data_collator = DataCollatorForSeq2Seq(
        tokenizer=tokenizer,
        model=model,
        padding=True,
        pad_to_multiple_of=8,
        label_pad_token_id=-100,
    )

    early_stopping = EarlyStoppingCallback(
        early_stopping_patience=CONFIG["training"]["early_stopping_patience"]
    )

    # FIX BUG #5 — NaNGuardCallback en premier pour stopper avant EarlyStopping
    nan_guard = NaNGuardCallback()

    return Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=validation_dataset,
        processing_class=tokenizer,
        data_collator=data_collator,
        callbacks=[nan_guard, early_stopping],
    )


def publish_training_artifacts() -> None:
    output_dir = Path(CONFIG["training"]["output_dir"])
    metadata = TrainingUtils.build_training_metadata(config=CONFIG)
    TrainingUtils.save_training_metadata(
        metadata=metadata,
        output_directory=output_dir,
    )
    logger.info("Training metadata saved.")


def train(publish_to_hf: bool = False):   # False par défaut en validation
    TrainingUtils.setup_logging(CONFIG["system"]["logging_level"])
    logger.info("Starting SFT validation run.")

    TrainingUtils.set_seed(CONFIG["system"]["seed"])

    wandb_run = TrainingUtils.initialize_wandb(config=CONFIG)

    tokenizer = TrainingTokenizerLoader.build(config=CONFIG)
    model = TrainingModelLoader.build(config=CONFIG)

    train_dataset, validation_dataset = prepare_datasets(tokenizer=tokenizer)

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
    TrainingUtils.finalize_wandb_run(wandb_run)

    logger.info("SFT validation run completed.")


if __name__ == "__main__":
    train()
