# medical-triage-agent-ai-poc/backend/app/training/dpo/train_dpo.py

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Dict, List, Tuple

import yaml
from datasets import Dataset, load_dataset

try:
    from trl import DPOTrainer, DPOConfig
except Exception:
    DPOTrainer = None
    DPOConfig = None

from backend.app.training.shared.training_model_loader import TrainingModelLoader  # noqa : E501
from backend.app.training.shared.training_tokenizer_loader import TrainingTokenizerLoader  # noqa : E501
from backend.app.training.shared.training_utils import TrainingUtils

logger = logging.getLogger(__name__)

CONFIG_PATH = Path(__file__).parent / "dpo_config.yaml"


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


def load_hf_dataset(dataset_repo: str, dataset_config: str, split: str) -> Dataset:  # noqa : E501
    return load_dataset(path=dataset_repo, name=dataset_config, split=split)


def load_dataset_source(split: str) -> Dataset:
    dataset_config = CONFIG["dataset"]
    hf_repo = dataset_config.get("hf_repo")
    hf_config = dataset_config.get("hf_config", "dpo")

    if hf_repo:
        return load_hf_dataset(hf_repo, hf_config, split)

    path_mapping = {
        "train": "train_path",
        "validation": "validation_path",
        "test": "test_path",
        "clinical_eval": "clinical_eval_path",
    }
    return load_jsonl_dataset(dataset_config[path_mapping[split]])


def build_dpo_sample(example: Dict) -> Dict:
    return {
        "prompt": example["prompt"],
        "chosen": example["chosen"],
        "rejected": example["rejected"],
    }


def prepare_datasets() -> Tuple[Dataset, Dataset]:
    train_dataset = load_dataset_source("train").map(build_dpo_sample)
    validation_dataset = load_dataset_source("validation").map(build_dpo_sample)  # noqa : E501
    return train_dataset, validation_dataset


def build_dpo_config():
    training_config = CONFIG["training"]
    return DPOConfig(
        output_dir=training_config["output_dir"],
        num_train_epochs=training_config["num_train_epochs"],
        per_device_train_batch_size=training_config["per_device_train_batch_size"],  # noqa : E501
        per_device_eval_batch_size=training_config["per_device_eval_batch_size"],  # noqa : E501
        gradient_accumulation_steps=training_config["gradient_accumulation_steps"],  # noqa : E501
        learning_rate=float(training_config["learning_rate"]),
        logging_steps=training_config["logging_steps"],
        eval_steps=training_config["eval_steps"],
        save_steps=training_config["save_steps"],
        save_total_limit=training_config["save_total_limit"],
        bf16=training_config.get("bf16", True),
        fp16=training_config.get("fp16", False),
        gradient_checkpointing=training_config["gradient_checkpointing"],
        report_to=training_config.get("report_to", ["wandb"]),
    )


def build_trainer(model, tokenizer, train_dataset, validation_dataset):
    if DPOTrainer is None:
        raise ImportError("TRL package is required for DPO training.")
    return DPOTrainer(
        model=model,
        args=build_dpo_config(),
        processing_class=tokenizer,
        train_dataset=train_dataset,
        eval_dataset=validation_dataset,
    )


def publish_training_artifacts() -> None:
    metadata = TrainingUtils.build_training_metadata(config=CONFIG)
    TrainingUtils.save_training_metadata(
        metadata=metadata,
        output_directory=CONFIG["training"]["output_dir"],
    )


def run_clinical_evaluation(model, tokenizer) -> None:
    logger.info("Clinical DPO evaluation integration pending.")


def train(publish_to_hf: bool = True):
    TrainingUtils.setup_logging(CONFIG["system"]["logging_level"])
    TrainingUtils.set_seed(CONFIG["system"]["seed"])

    wandb_run = TrainingUtils.initialize_wandb(config=CONFIG)

    tokenizer = TrainingTokenizerLoader.build(config=CONFIG)
    model = TrainingModelLoader.build(config=CONFIG)

    train_dataset, validation_dataset = prepare_datasets()

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


if __name__ == "__main__":
    train()
