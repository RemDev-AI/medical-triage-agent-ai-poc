# medical-triage-agent-ai-poc/backend/app/training/sft/train_sft.py

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Dict, List, Tuple

import yaml
from datasets import Dataset, load_dataset
from transformers import (
    EarlyStoppingCallback,
    Trainer,
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

CONFIG_PATH = Path(__file__).parent / "sft_config.yaml"


def load_config() -> Dict:
    with open(CONFIG_PATH, "r", encoding="utf-8") as file:
        return yaml.safe_load(file)


CONFIG = load_config()


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
        return load_hf_dataset(
            dataset_repo=hf_repo,
            dataset_config=hf_config,
            split=split,
        )

    path_mapping = {
        "train": "train_path",
        "validation": "validation_path",
        "test": "test_path",
        "clinical_eval": "clinical_eval_path",
    }

    return load_jsonl_dataset(
        dataset_config[path_mapping[split]]
    )


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


def tokenize_function(
    examples,
    tokenizer,
    max_length: int,
):
    outputs = tokenizer(
        examples["text"],
        truncation=True,
        padding="max_length",
        max_length=max_length,
    )

    outputs["labels"] = outputs["input_ids"].copy()

    return outputs


def prepare_datasets(
    tokenizer,
) -> Tuple[Dataset, Dataset]:

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
            x,
            tokenizer,
            dataset_config["max_sequence_length"],
        ),
        batched=True,
    )

    validation_dataset = validation_dataset.map(
        lambda x: tokenize_function(
            x,
            tokenizer,
            dataset_config["max_sequence_length"],
        ),
        batched=True,
    )

    return train_dataset, validation_dataset


def build_training_arguments() -> TrainingArguments:

    training_config = CONFIG["training"]

    training_args = {
        "output_dir": training_config["output_dir"],
        "num_train_epochs": training_config["num_train_epochs"],
        "per_device_train_batch_size": training_config[
            "per_device_train_batch_size"
        ],
        "per_device_eval_batch_size": training_config[
            "per_device_eval_batch_size"
        ],
        "gradient_accumulation_steps": training_config[
            "gradient_accumulation_steps"
        ],
        "learning_rate": float(
            training_config["learning_rate"]
        ),
        "weight_decay": float(
            training_config["weight_decay"]
        ),
        "warmup_ratio": float(
            training_config["warmup_ratio"]
        ),
        "logging_steps": training_config["logging_steps"],
        "eval_steps": training_config["eval_steps"],
        "save_steps": training_config["save_steps"],
        "save_total_limit": training_config[
            "save_total_limit"
        ],
        "eval_strategy": training_config[
            "evaluation_strategy"
        ],
        "save_strategy": training_config[
            "save_strategy"
        ],
        "load_best_model_at_end": training_config[
            "load_best_model_at_end"
        ],
        "metric_for_best_model": training_config[
            "metric_for_best_model"
        ],
        "greater_is_better": training_config[
            "greater_is_better"
        ],
        "gradient_checkpointing": training_config[
            "gradient_checkpointing"
        ],
        "lr_scheduler_type": training_config[
            "lr_scheduler_type"
        ],
        "max_grad_norm": float(
            training_config["max_grad_norm"]
        ),
        "report_to": training_config.get(
            "report_to",
            ["wandb"],
        ),
        "remove_unused_columns": True,
    }

    training_args = apply_precision_arguments(
        training_args
    )

    return TrainingArguments(**training_args)


def build_trainer(
    model,
    tokenizer,
    train_dataset,
    validation_dataset,
):

    training_args = build_training_arguments()

    early_stopping = EarlyStoppingCallback(
        early_stopping_patience=CONFIG["training"][
            "early_stopping_patience"
        ]
    )

    return Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=validation_dataset,
        processing_class=tokenizer,
        callbacks=[early_stopping],
    )


def publish_training_artifacts() -> None:

    output_dir = Path(
        CONFIG["training"]["output_dir"]
    )

    metadata = (
        TrainingUtils.build_training_metadata(
            config=CONFIG,
        )
    )

    TrainingUtils.save_training_metadata(
        metadata=metadata,
        output_directory=output_dir,
    )

    logger.info(
        "Training metadata saved."
    )


def run_clinical_evaluation(
    model,
    tokenizer,
) -> None:
    """
    Future integration point.

    from backend.app.training.evaluation.clinical_eval_runner import (
        ClinicalEvalRunner
    )
    """
    logger.info(
        "Clinical evaluation integration pending."
    )


def train(
    publish_to_hf: bool = True,
):

    TrainingUtils.setup_logging(
        CONFIG["system"]["logging_level"]
    )

    logger.info(
        "Starting SFT training."
    )

    TrainingUtils.set_seed(
        CONFIG["system"]["seed"]
    )

    wandb_run = (
        TrainingUtils.initialize_wandb(
            config=CONFIG,
        )
    )

    tokenizer = (
        TrainingTokenizerLoader.build(
            config=CONFIG,
        )
    )

    model = (
        TrainingModelLoader.build(
            config=CONFIG,
        )
    )

    train_dataset, validation_dataset = (
        prepare_datasets(
            tokenizer=tokenizer,
        )
    )

    trainer = build_trainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=train_dataset,
        validation_dataset=validation_dataset,
    )

    trainer.train(
        resume_from_checkpoint=CONFIG["training"].get(
            "resume_from_checkpoint"
        )
    )

    trainer.save_model()

    tokenizer.save_pretrained(
        CONFIG["training"]["output_dir"]
    )

    publish_training_artifacts()

    run_clinical_evaluation(
        model=model,
        tokenizer=tokenizer,
    )

    TrainingUtils.finalize_wandb_run(
        wandb_run
    )

    logger.info(
        "SFT training completed."
    )


if __name__ == "__main__":
    train()
