# medical-triage-agent-ai-poc/backend/app/training/dpo/train_dpo.py

from __future__ import annotations

import json
import logging
import random
from pathlib import Path
from typing import Dict
from typing import List
from typing import Tuple

# import mlflow
import torch
import wandb
import yaml

from datasets import Dataset
from datasets import load_dataset

from transformers import AutoModelForCausalLM, AutoTokenizer, EarlyStoppingCallback, TrainingArguments  # noqa : E501

from trl import DPOTrainer

from backend.app.training.lora.peft_setup import setup_peft_model

from backend.app.training.modal.modal_utils import build_training_metadata, download_latest_model_snapshot, save_training_metadata, upload_final_model  # noqa : E501

logger = logging.getLogger(__name__)

CONFIG_PATH = (
    Path(__file__).parent / "dpo_config.yaml"
)

LOCAL_SFT_MODEL_DIR = (
    Path("artifacts")
    / "sft_model"
)


def load_config() -> Dict:

    with open(
        CONFIG_PATH,
        "r",
        encoding="utf-8",
    ) as file:
        return yaml.safe_load(file)


CONFIG = load_config()


# ============================================================
# LOGGING
# ============================================================


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

    tracking = CONFIG["tracking"]

    # mlflow.set_experiment(
    #     tracking[
    #         "mlflow_experiment_name"
    #     ]
    # )

    wandb.init(
        project=tracking[
            "wandb_project"
        ],
        name=tracking[
            "wandb_run_name"
        ],
    )


# ============================================================
# DATASETS
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

    dataset_config = CONFIG[
        "dataset"
    ]

    hf_repo = dataset_config.get(
        "hf_repo"
    )

    hf_config = dataset_config.get(
        "hf_config",
        "dpo",
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
# SAFETY FILTERS
# ============================================================


def detect_hallucination(
    text: str,
) -> bool:

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

    safe_records = []

    for sample in dataset:

        chosen = sample["chosen"]
        rejected = sample["rejected"]

        chosen_unsafe = (
            detect_hallucination(
                chosen
            )
            or
            detect_dangerous_recommendation(
                chosen
            )
        )

        rejected_unsafe = (
            detect_hallucination(
                rejected
            )
            or
            detect_dangerous_recommendation(
                rejected
            )
        )

        if not chosen_unsafe:
            safe_records.append(
                sample
            )

        if rejected_unsafe:
            logger.warning(
                "Unsafe rejected response detected."
            )

    logger.info(
        "Clinical safety filtering completed. "
        "Safe samples retained: %s",
        len(safe_records),
    )

    return Dataset.from_list(
        safe_records
    )


def prepare_datasets() -> Tuple[
    Dataset,
    Dataset,
]:

    logger.info(
        "Preparing DPO datasets..."
    )

    train_dataset = load_dataset_source(
        "train"
    )

    validation_dataset = load_dataset_source(
        "validation"
    )

    logger.info(
        "Raw train samples: %s",
        len(train_dataset),
    )

    logger.info(
        "Raw validation samples: %s",
        len(validation_dataset),
    )

    train_dataset = (
        validate_clinical_safety(
            train_dataset
        )
    )

    validation_dataset = (
        validate_clinical_safety(
            validation_dataset
        )
    )

    logger.info(
        "Filtered train samples: %s",
        len(train_dataset),
    )

    logger.info(
        "Filtered validation samples: %s",
        len(validation_dataset),
    )

    return (
        train_dataset,
        validation_dataset,
    )


# ============================================================
# MODEL
# ============================================================


def resolve_base_model_path() -> str:

    local_model = CONFIG[
        "model"
    ].get(
        "sft_model_path"
    )

    if local_model:
        return local_model

    LOCAL_SFT_MODEL_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    download_latest_model_snapshot(
        LOCAL_SFT_MODEL_DIR
    )

    return str(
        LOCAL_SFT_MODEL_DIR
    )


def load_tokenizer():

    tokenizer_path = (
        resolve_base_model_path()
    )

    tokenizer = (
        AutoTokenizer
        .from_pretrained(
            tokenizer_path,
            trust_remote_code=CONFIG["model"][
                "trust_remote_code"
            ],
        )
    )

    tokenizer.pad_token = (
        tokenizer.eos_token
    )

    return tokenizer


def load_model():

    model_path = (
        resolve_base_model_path()
    )

    logger.info(
        "Loading SFT model..."
    )

    model = (
        AutoModelForCausalLM
        .from_pretrained(
            model_path,
            trust_remote_code=CONFIG["model"][
                "trust_remote_code"
            ],
            device_map=CONFIG["model"][
                "device_map"
            ],
            torch_dtype=torch.bfloat16,
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
# TRAINER
# ============================================================


def build_training_arguments():

    training = CONFIG[
        "training"
    ]

    return TrainingArguments(
        output_dir=training["output_dir"],

        num_train_epochs=training[
            "num_train_epochs"
        ],

        per_device_train_batch_size=training[
            "per_device_train_batch_size"
        ],

        per_device_eval_batch_size=training[
            "per_device_eval_batch_size"
        ],

        gradient_accumulation_steps=training[
            "gradient_accumulation_steps"
        ],

        learning_rate=float(
            training[
                "learning_rate"
            ]
        ),

        warmup_ratio=float(
            training[
                "warmup_ratio"
            ]
        ),

        logging_steps=training[
            "logging_steps"
        ],

        eval_steps=training[
            "eval_steps"
        ],

        save_steps=training[
            "save_steps"
        ],

        save_total_limit=training[
            "save_total_limit"
        ],

        evaluation_strategy=training[
            "evaluation_strategy"
        ],

        save_strategy=training[
            "save_strategy"
        ],

        load_best_model_at_end=training[
            "load_best_model_at_end"
        ],

        metric_for_best_model=training[
            "metric_for_best_model"
        ],

        greater_is_better=training[
            "greater_is_better"
        ],

        bf16=training["bf16"],

        fp16=training["fp16"],

        gradient_checkpointing=training[
            "gradient_checkpointing"
        ],

        lr_scheduler_type=training[
            "lr_scheduler_type"
        ],

        max_grad_norm=float(
            training[
                "max_grad_norm"
            ]
        ),

        report_to=training[
            "report_to"
        ],
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
            early_stopping_patience=CONFIG["training"][
                "early_stopping_patience"
            ]
        )
    )

    return DPOTrainer(
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
        max_length=CONFIG["dpo"][
            "max_length"
        ],
        callbacks=[
            early_stopping
        ],
    )


# ============================================================
# PUBLISH
# ============================================================


def publish_training_artifacts():

    output_dir = Path(
        CONFIG["training"][
            "output_dir"
        ]
    )

    metadata = (
        build_training_metadata(
            training_type="dpo",
            base_model=resolve_base_model_path(),
            dataset_name=CONFIG["dataset"].get(
                "hf_dataset_repo",
                "local_dataset",
            ),
            extra={
                "dataset_config": "dpo",
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
    #     stage="dpo",
    # )


# ============================================================
# MAIN
# ============================================================


def train(
    publish_to_hf: bool = True,
):

    setup_logging()

    logger.info(
        "Starting DPO training..."
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
        prepare_datasets()
    )

    trainer = build_trainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=train_dataset,
        validation_dataset=validation_dataset,
    )

    logger.info(
        "Launching DPO training..."
    )

    trainer.train(
        resume_from_checkpoint=CONFIG["training"][
            "resume_from_checkpoint"
        ]
    )

    logger.info(
        "Saving aligned model..."
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
        "DPO training completed."
    )

    wandb.finish()


if __name__ == "__main__":
    train()
