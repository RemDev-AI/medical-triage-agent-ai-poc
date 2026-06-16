# medical-triage-agent-ai-poc/backend/app/training/dpo/train_dpo.py

"""
DPO training pipeline for medical triage alignment.

Features:
- Direct Preference Optimization
- chosen/rejected alignment
- hallucination reduction
- dangerous recommendation detection
- MLflow tracking
- Weights & Biases integration
- checkpoint management
"""

import json
import logging
import random
from pathlib import Path
from typing import Dict, List

import mlflow
import torch
import wandb
import yaml
from datasets import Dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    EarlyStoppingCallback,
    TrainingArguments,
)

from trl import DPOTrainer

from backend.app.training.lora.peft_setup import setup_peft_model

logger = logging.getLogger(__name__)

CONFIG_PATH = (
    Path(__file__).parent / "dpo_config.yaml"
)


def load_config() -> Dict:
    """
    Load YAML configuration.
    """

    with open(CONFIG_PATH, "r", encoding="utf-8") as file:
        return yaml.safe_load(file)


CONFIG = load_config()


def setup_logging():
    """
    Configure logging.
    """

    logging.basicConfig(
        level=CONFIG["system"]["logging_level"],
        format=(
            "%(asctime)s | %(levelname)s "
            "| %(name)s | %(message)s"
        ),
    )


def set_seed(seed: int):
    """
    Ensure reproducibility.
    """

    random.seed(seed)

    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def initialize_tracking():
    """
    Initialize experiment tracking.
    """

    tracking = CONFIG["tracking"]

    mlflow.set_experiment(
        tracking["mlflow_experiment_name"]
    )

    wandb.init(
        project=tracking["wandb_project"],
        name=tracking["wandb_run_name"],
    )


def load_jsonl_dataset(path: str) -> Dataset:
    """
    Load DPO dataset.

    Expected format:
    {
        "prompt": "...",
        "chosen": "...",
        "rejected": "..."
    }
    """

    records: List[Dict] = []

    with open(path, "r", encoding="utf-8") as file:
        for line in file:
            records.append(json.loads(line))

    return Dataset.from_list(records)


def detect_hallucination(text: str) -> bool:
    """
    Detect hallucination patterns.
    """

    keywords = CONFIG["safety"][
        "hallucination_keywords"
    ]

    text = text.lower()

    return any(
        keyword.lower() in text
        for keyword in keywords
    )


def detect_dangerous_recommendation(
    text: str,
) -> bool:
    """
    Detect dangerous medical advice.
    """

    keywords = CONFIG["safety"][
        "dangerous_keywords"
    ]

    text = text.lower()

    return any(
        keyword.lower() in text
        for keyword in keywords
    )


def validate_clinical_safety(
    dataset: Dataset,
) -> Dataset:
    """
    Remove unsafe samples.
    """

    safe_records = []

    for sample in dataset:

        chosen = sample["chosen"]
        rejected = sample["rejected"]

        chosen_unsafe = (
            detect_hallucination(chosen)
            or detect_dangerous_recommendation(
                chosen
            )
        )

        rejected_unsafe = (
            detect_hallucination(rejected)
            or detect_dangerous_recommendation(
                rejected
            )
        )

        if not chosen_unsafe:
            safe_records.append(sample)

        if rejected_unsafe:
            logger.warning(
                "Unsafe rejected response detected."
            )

    logger.info(
        "Clinical safety filtering completed. "
        "Safe samples retained: %s",
        len(safe_records),
    )

    return Dataset.from_list(safe_records)


def load_tokenizer():
    """
    Load tokenizer.
    """

    model_config = CONFIG["model"]

    tokenizer = AutoTokenizer.from_pretrained(
        model_config["model_name"],
        trust_remote_code=model_config[
            "trust_remote_code"
        ],
    )

    tokenizer.pad_token = tokenizer.eos_token

    return tokenizer


def load_model():
    """
    Load Qwen model and inject LoRA.
    """

    model_config = CONFIG["model"]

    logger.info("Loading base model...")

    model = AutoModelForCausalLM.from_pretrained(
        model_config["model_name"],
        trust_remote_code=model_config[
            "trust_remote_code"
        ],
        device_map=model_config["device_map"],
        torch_dtype=torch.bfloat16,
    )

    logger.info("Injecting LoRA adapters...")

    model = setup_peft_model(model)

    return model


def prepare_datasets():
    """
    Load and validate datasets.
    """

    dataset_config = CONFIG["dataset"]

    train_dataset = load_jsonl_dataset(
        dataset_config["train_path"]
    )

    validation_dataset = load_jsonl_dataset(
        dataset_config["validation_path"]
    )

    train_dataset = validate_clinical_safety(
        train_dataset
    )

    validation_dataset = (
        validate_clinical_safety(
            validation_dataset
        )
    )

    return train_dataset, validation_dataset


def build_training_arguments():
    """
    Build Hugging Face training arguments.
    """

    training_config = CONFIG["training"]

    return TrainingArguments(
        output_dir=training_config["output_dir"],

        num_train_epochs=training_config[
            "num_train_epochs"
        ],

        per_device_train_batch_size=training_config[
            "per_device_train_batch_size"
        ],

        per_device_eval_batch_size=training_config[
            "per_device_eval_batch_size"
        ],

        gradient_accumulation_steps=training_config[
            "gradient_accumulation_steps"
        ],

        learning_rate=float(
            training_config["learning_rate"]
        ),

        warmup_ratio=float(
            training_config["warmup_ratio"]
        ),

        logging_steps=training_config[
            "logging_steps"
        ],

        eval_steps=training_config[
            "eval_steps"
        ],

        save_steps=training_config[
            "save_steps"
        ],

        save_total_limit=training_config[
            "save_total_limit"
        ],

        evaluation_strategy=training_config[
            "evaluation_strategy"
        ],

        save_strategy=training_config[
            "save_strategy"
        ],

        load_best_model_at_end=training_config[
            "load_best_model_at_end"
        ],

        metric_for_best_model=training_config[
            "metric_for_best_model"
        ],

        greater_is_better=training_config[
            "greater_is_better"
        ],

        bf16=training_config["bf16"],

        fp16=training_config["fp16"],

        gradient_checkpointing=training_config[
            "gradient_checkpointing"
        ],

        lr_scheduler_type=training_config[
            "lr_scheduler_type"
        ],

        max_grad_norm=float(
            training_config["max_grad_norm"]
        ),

        report_to=training_config["report_to"],
    )


def build_trainer(
    model,
    tokenizer,
    train_dataset,
    validation_dataset,
):
    """
    Build DPO trainer.
    """

    training_args = build_training_arguments()

    early_stopping = EarlyStoppingCallback(
        early_stopping_patience=CONFIG[
            "training"
        ]["early_stopping_patience"]
    )

    trainer = DPOTrainer(
        model=model,

        ref_model=None,

        args=training_args,

        beta=CONFIG["dpo"]["beta"],

        train_dataset=train_dataset,

        eval_dataset=validation_dataset,

        tokenizer=tokenizer,

        max_prompt_length=CONFIG["dpo"][
            "max_prompt_length"
        ],

        max_length=CONFIG["dpo"]["max_length"],

        callbacks=[early_stopping],
    )

    return trainer


def train():
    """
    Main DPO training pipeline.
    """

    setup_logging()

    logger.info(
        "Starting DPO training pipeline..."
    )

    set_seed(CONFIG["system"]["seed"])

    initialize_tracking()

    tokenizer = load_tokenizer()

    model = load_model()

    train_dataset, validation_dataset = (
        prepare_datasets()
    )

    trainer = build_trainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=train_dataset,
        validation_dataset=validation_dataset,
    )

    logger.info("Launching DPO training...")

    trainer.train(
        resume_from_checkpoint=CONFIG[
            "training"
        ]["resume_from_checkpoint"]
    )

    logger.info("Saving DPO aligned model...")

    trainer.save_model()

    tokenizer.save_pretrained(
        CONFIG["training"]["output_dir"]
    )

    logger.info("DPO training completed.")

    wandb.finish()


if __name__ == "__main__":
    train()
