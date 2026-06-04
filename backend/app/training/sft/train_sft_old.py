# medical-triage-agent-ai-poc/backend/app/training/sft/train_sft.py

"""
SFT training pipeline for Qwen3 medical triage model.

Features:
- LoRA fine-tuning
- MLflow tracking
- Weights & Biases tracking
- checkpoint management
- resume training
- early stopping
- evaluation
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
    Trainer,
    TrainingArguments,
)

from training.lora.peft_setup import setup_peft_model

logger = logging.getLogger(__name__)


CONFIG_PATH = (
    Path(__file__).parent / "sft_config.yaml"
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


def load_jsonl_dataset(path: str) -> Dataset:
    """
    Load JSONL dataset.

    Expected format:
    {
        "instruction": "...",
        "response": "..."
    }
    """

    records: List[Dict] = []

    with open(path, "r", encoding="utf-8") as file:
        for line in file:
            records.append(json.loads(line))

    return Dataset.from_list(records)


def build_prompt(example: Dict) -> Dict:
    """
    Build medical triage instruction prompt.
    """

    prompt = (
        "<|system|>\n"
        "You are a medical triage assistant.\n"
        "Provide clinically safe recommendations.\n"
        "<|user|>\n"
        f"{example['instruction']}\n"
        "<|assistant|>\n"
        f"{example['response']}"
    )

    return {
        "text": prompt,
    }


def tokenize_function(
    examples,
    tokenizer,
    max_length: int,
):
    """
    Tokenize prompts.
    """

    outputs = tokenizer(
        examples["text"],
        truncation=True,
        padding="max_length",
        max_length=max_length,
    )

    outputs["labels"] = outputs["input_ids"].copy()

    return outputs


def initialize_tracking():
    """
    Initialize MLflow and Weights & Biases.
    """

    tracking_config = CONFIG["tracking"]

    mlflow.set_experiment(
        tracking_config["mlflow_experiment_name"]
    )

    wandb.init(
        project=tracking_config["wandb_project"],
        name=tracking_config["wandb_run_name"],
    )


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
    Load base Qwen model.
    """

    model_config = CONFIG["model"]

    logger.info("Loading base model...")

    model = AutoModelForCausalLM.from_pretrained(
        model_config["model_name"],
        device_map=model_config["device_map"],
        trust_remote_code=model_config[
            "trust_remote_code"
        ],
        torch_dtype=torch.bfloat16,
    )

    logger.info("Injecting LoRA adapters...")

    model = setup_peft_model(model)

    return model


def prepare_datasets(tokenizer):
    """
    Load and tokenize datasets.
    """

    dataset_config = CONFIG["dataset"]

    train_dataset = load_jsonl_dataset(
        dataset_config["train_path"]
    )

    validation_dataset = load_jsonl_dataset(
        dataset_config["validation_path"]
    )

    train_dataset = train_dataset.map(build_prompt)

    validation_dataset = validation_dataset.map(
        build_prompt
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


def build_training_arguments():
    """
    Build Hugging Face TrainingArguments.
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

        weight_decay=float(
            training_config["weight_decay"]
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
    Build Trainer object.
    """

    training_args = build_training_arguments()

    early_stopping = EarlyStoppingCallback(
        early_stopping_patience=CONFIG[
            "training"
        ]["early_stopping_patience"]
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=validation_dataset,
        tokenizer=tokenizer,
        callbacks=[early_stopping],
    )

    return trainer


def train():
    """
    Main SFT training pipeline.
    """

    setup_logging()

    logger.info("Starting SFT training pipeline...")

    set_seed(CONFIG["system"]["seed"])

    initialize_tracking()

    tokenizer = load_tokenizer()

    model = load_model()

    train_dataset, validation_dataset = (
        prepare_datasets(tokenizer)
    )

    trainer = build_trainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=train_dataset,
        validation_dataset=validation_dataset,
    )

    logger.info("Launching training...")

    trainer.train(
        resume_from_checkpoint=CONFIG[
            "training"
        ]["resume_from_checkpoint"]
    )

    logger.info("Saving final model...")

    trainer.save_model()

    tokenizer.save_pretrained(
        CONFIG["training"]["output_dir"]
    )

    logger.info("Training completed.")

    wandb.finish()


if __name__ == "__main__":
    train()
