# medical-triage-agent-ai-poc/backend/app/training/sft/train_sft.py

from __future__ import annotations

import json
import logging
import random
from pathlib import Path
from typing import Dict
from typing import List
from typing import Optional  # noqa : F401
from typing import Tuple

# import mlflow
import torch
import wandb
import yaml

from datasets import Dataset
from datasets import load_dataset

from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    EarlyStoppingCallback,
    Trainer,
    TrainingArguments,
)

from backend.app.training.lora.peft_setup import setup_peft_model

from backend.app.training.modal.modal_utils import ( # noqa : F401
    build_training_metadata,
    save_training_metadata,
    upload_final_model,
)

logger = logging.getLogger(__name__)

CONFIG_PATH = (
    Path(__file__).parent / "sft_config.yaml"
)


def load_config() -> Dict:
    with open(
        CONFIG_PATH,
        "r",
        encoding="utf-8",
    ) as file:
        return yaml.safe_load(file)


CONFIG = load_config()


def setup_logging():
    logging.basicConfig(
        level=CONFIG["system"]["logging_level"],
        format=(
            "%(asctime)s | %(levelname)s "
            "| %(name)s | %(message)s"
        ),
    )


def set_seed(seed: int):
    random.seed(seed)

    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def initialize_tracking():
    tracking_config = CONFIG["tracking"]

    # mlflow.set_experiment(
    #     tracking_config[
    #         "mlflow_experiment_name"
    #     ]
    # )

    wandb.init(
        project=tracking_config[
            "wandb_project"
        ],
        name=tracking_config[
            "wandb_run_name"
        ],
    )


# ============================================================
# DATASET LOADERS
# ============================================================


def load_jsonl_dataset(
    path: str,
) -> Dataset:

    records: List[Dict] = []

    with open(
        path,
        "r",
        encoding="utf-8",
    ) as file:

        for line in file:
            records.append(
                json.loads(line)
            )

    return Dataset.from_list(records)


def load_hf_dataset(
    dataset_repo: str,
    dataset_config: str,
    split: str,
) -> Dataset:
    """
    Load dataset from Hugging Face.

    Supported configs:
        - sft
        - dpo

    Supported splits:
        - train
        - validation
        - test
        - clinical_eval
    """

    supported_configs = {
        "sft",
        "dpo",
    }

    supported_splits = {
        "train",
        "validation",
        "test",
        "clinical_eval",
    }

    if dataset_config not in supported_configs:
        raise ValueError(
            f"Unsupported dataset config: "
            f"{dataset_config}. "
            f"Supported configs: "
            f"{sorted(supported_configs)}"
        )

    if split not in supported_splits:
        raise ValueError(
            f"Unsupported dataset split: "
            f"{split}. "
            f"Supported splits: "
            f"{sorted(supported_splits)}"
        )

    logger.info(
        "Loading HF dataset "
        "(repo=%s, config=%s, split=%s)",
        dataset_repo,
        dataset_config,
        split,
    )

    dataset = load_dataset(
        path=dataset_repo,
        name=dataset_config,
        split=split,
    )

    logger.info(
        "Loaded %s samples from %s/%s",
        len(dataset),
        dataset_config,
        split,
    )

    return dataset


def load_dataset_source(
    split: str,
) -> Dataset:
    """
    Dynamic dataset loading.

    Priority:
    1. Hugging Face Dataset
    2. Local JSONL
    """

    dataset_config = CONFIG["dataset"]

    hf_repo = dataset_config.get(
        "hf_repo"
    )

    hf_config = dataset_config.get(
        "hf_config",
        "sft",
    )

    if hf_repo:
        
        logger.info(
            "Loading HF dataset "
            "(repo=%s, config=%s, split=%s)",
            hf_repo,
            hf_config,
            split,
        )

        return load_hf_dataset(
            dataset_repo=hf_repo,
            dataset_config=hf_config,
            split=split,
        )

    path_key = (
        "train_path"
        if split == "train"
        else "validation_path"
    )

    return load_jsonl_dataset(
        dataset_config[path_key]
    )


# ============================================================
# PROMPTS
# ============================================================


def build_prompt(
    example: Dict,
) -> Dict:

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

    outputs = tokenizer(
        examples["text"],
        truncation=True,
        padding="max_length",
        max_length=max_length,
    )

    outputs["labels"] = (
        outputs["input_ids"].copy()
    )

    return outputs


# ============================================================
# MODEL
# ============================================================


def load_tokenizer():

    model_config = CONFIG["model"]

    tokenizer = (
        AutoTokenizer.from_pretrained(
            model_config["model_name"],
            trust_remote_code=model_config[
                "trust_remote_code"
            ],
        )
    )

    if tokenizer.pad_token is None:
        tokenizer.pad_token = (
            tokenizer.eos_token
        )

    return tokenizer


def load_model():

    model_config = CONFIG["model"]

    logger.info(
        "Loading base model..."
    )

    has_cuda = (
        torch.cuda.is_available()
    )

    dtype = (
        torch.bfloat16
        if has_cuda
        else torch.float32
    )

    device_map = (
        model_config["device_map"]
        if has_cuda
        else None
    )

    logger.info(
        "CUDA available: %s",
        has_cuda,
    )

    logger.info(
        "Using dtype: %s",
        dtype,
    )

    model = (
        AutoModelForCausalLM
        .from_pretrained(
            model_config[
                "model_name"
            ],
            device_map=device_map,
            trust_remote_code=model_config[
                "trust_remote_code"
            ],
            torch_dtype=dtype,
        )
    )

    logger.info(
        "Injecting LoRA adapters..."
    )

    model = setup_peft_model(
        model
    )

    return model


# ============================================================
# DATASET PREPARATION
# ============================================================


def prepare_datasets(
    tokenizer,
) -> Tuple[
    Dataset,
    Dataset,
]:

    dataset_config = CONFIG[
        "dataset"
    ]

    train_dataset = (
        load_dataset_source(
            "train"
        )
    )

    validation_dataset = (
        load_dataset_source(
            "validation"
        )
    )

    train_dataset = (
        train_dataset.map(
            build_prompt,
            remove_columns=train_dataset.column_names,
        )
    )

    validation_dataset = (
        validation_dataset.map(
            build_prompt,
            remove_columns=validation_dataset.column_names,
        )
    )

    train_dataset = (
        train_dataset.map(
            lambda x:
            tokenize_function(
                x,
                tokenizer,
                dataset_config[
                    "max_sequence_length"
                ],
            ),
            batched=True,
        )
    )

    validation_dataset = (
        validation_dataset.map(
            lambda x:
            tokenize_function(
                x,
                tokenizer,
                dataset_config[
                    "max_sequence_length"
                ],
            ),
            batched=True,
        )
    )

    return (
        train_dataset,
        validation_dataset,
    )


# ============================================================
# TRAINER
# ============================================================

def build_training_arguments():

    training_config = CONFIG[
        "training"
    ]

    has_cuda = (
        torch.cuda.is_available()
    )

    bf16 = (
        training_config["bf16"]
        and has_cuda
    )

    fp16 = (
        training_config["fp16"]
        and has_cuda
    )

    return TrainingArguments(
        output_dir=training_config[
            "output_dir"
        ],

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
            training_config[
                "learning_rate"
            ]
        ),

        weight_decay=float(
            training_config[
                "weight_decay"
            ]
        ),

        warmup_ratio=float(
            training_config[
                "warmup_ratio"
            ]
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

        eval_strategy=training_config[
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

        bf16=bf16,

        fp16=fp16,

        gradient_checkpointing=training_config[
            "gradient_checkpointing"
        ],

        lr_scheduler_type=training_config[
            "lr_scheduler_type"
        ],

        max_grad_norm=float(
            training_config[
                "max_grad_norm"
            ]
        ),

        report_to=training_config[
            "report_to"
        ],

        remove_unused_columns=True,
    )


def build_trainer(
    model,
    tokenizer,
    train_dataset,
    validation_dataset,
):

    training_args = (
        build_training_arguments()
    )

    early_stopping = (
        EarlyStoppingCallback(
            early_stopping_patience=CONFIG[
                "training"
            ][
                "early_stopping_patience"
            ]
        )
    )

    return Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=validation_dataset,
        processing_class=tokenizer,
        callbacks=[
            early_stopping
        ],
    )


# ============================================================
# SAVE / PUBLISH
# ============================================================


def publish_training_artifacts():

    output_dir = Path(
        CONFIG["training"][
            "output_dir"
        ]
    )

    metadata = (
        build_training_metadata(
            training_type="sft",
            base_model=CONFIG["model"][
                "model_name"
            ],
            dataset_name=CONFIG["dataset"].get(
                "hf_dataset_repo",
                "local_dataset",
            ),
            extra={
                "output_dir":
                str(output_dir),
            },
        )
    )

    save_training_metadata(
        metadata,
        output_dir
        / "training_metadata.json",
    )

    # upload_final_model(
    #     model_path=output_dir,
    #     stage="sft",
    # )


# ============================================================
# MAIN
# ============================================================


def train(
    publish_to_hf: bool = True,
):

    setup_logging()

    logger.info(
        "Starting SFT training..."
    )

    set_seed(
        CONFIG["system"]["seed"]
    )

    initialize_tracking()

    tokenizer = (
        load_tokenizer()
    )

    model = load_model()

    train_dataset, validation_dataset = (
        prepare_datasets(
            tokenizer
        )
    )

    trainer = build_trainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=train_dataset,
        validation_dataset=validation_dataset,
    )

    logger.info(
        "Launching training..."
    )

    trainer.train(
        resume_from_checkpoint=CONFIG["training"][
            "resume_from_checkpoint"
        ]
    )

    logger.info(
        "Saving model..."
    )

    trainer.save_model()

    tokenizer.save_pretrained(
        CONFIG["training"][
            "output_dir"
        ]
    )

    if publish_to_hf:

        publish_training_artifacts()

    logger.info(
        "Training completed."
    )

    wandb.finish()


if __name__ == "__main__":
    train()
