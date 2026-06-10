# medical-triage-agent-ai-poc/backend/app/training/modal/modal_sft_runner.py

from __future__ import annotations

import logging
from datetime import datetime
from pathlib import Path

import modal

from app.training.modal.modal_config import (
    app,
    config,
    hf_secret,
    mlflow_secret,
    training_image,
    training_volume,
    wandb_secret,
)
from app.training.modal.modal_utils import (
    build_training_metadata,
    commit_volume,
    download_project_dataset,
    save_training_metadata,
    upload_final_model,
)

logger = logging.getLogger(__name__)

SFT_OUTPUT_DIR = (
    Path(config.checkpoint_dir)
    / "sft"
)

METADATA_FILE = (
    SFT_OUTPUT_DIR
    / "training_metadata.json"
)


@app.function(
    image=training_image,
    gpu=modal.gpu.A100(
        size="40GB"
    ),
    cpu=config.cpu,
    memory=config.memory_mb,
    timeout=config.timeout,
    volumes={
        config.checkpoint_dir: training_volume,
    },
    secrets=[
        hf_secret,
        wandb_secret,
        mlflow_secret,
    ],
)
def run_sft_training() -> dict:
    """
    Execute complete SFT pipeline on Modal.
    """

    logger.info(
        "Starting Modal SFT training..."
    )

    logger.info(
        "Downloading dataset from Hugging Face..."
    )

    train_dataset = download_project_dataset(
        split="train"
    )

    validation_dataset = (
        download_project_dataset(
            split="validation"
        )
    )

    logger.info(
        "Train samples: %s",
        len(train_dataset),
    )

    logger.info(
        "Validation samples: %s",
        len(validation_dataset),
    )

    from training.sft.train_sft import (
        CONFIG,
        build_trainer,
        initialize_tracking,
        load_model,
        load_tokenizer,
        prepare_datasets,
        set_seed,
        setup_logging,
    )

    setup_logging()

    set_seed(
        CONFIG["system"]["seed"]
    )

    initialize_tracking()

    tokenizer = load_tokenizer()

    model = load_model()

    train_dataset_local, validation_dataset_local = (
        prepare_datasets(
            tokenizer
        )
    )

    trainer = build_trainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=train_dataset_local,
        validation_dataset=validation_dataset_local,
    )

    logger.info(
        "Launching SFT training..."
    )

    trainer.train(
        resume_from_checkpoint=CONFIG["training"][
            "resume_from_checkpoint"
        ]
    )

    logger.info(
        "Saving SFT model..."
    )

    SFT_OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    trainer.save_model(
        str(SFT_OUTPUT_DIR)
    )

    tokenizer.save_pretrained(
        str(SFT_OUTPUT_DIR)
    )

    metadata = (
        build_training_metadata(
            training_type="sft",
            base_model=CONFIG["model"][
                "model_name"
            ],
            dataset_name=config.dataset_repo,
            extra={
                "training_date":
                datetime.utcnow().isoformat(),
                "output_dir":
                str(SFT_OUTPUT_DIR),
                "modal_gpu":
                config.gpu_type,
            },
        )
    )

    save_training_metadata(
        metadata,
        METADATA_FILE,
    )

    commit_volume()

    logger.info(
        "Uploading model to Hugging Face..."
    )

    upload_final_model(
        model_path=SFT_OUTPUT_DIR,
        stage="sft",
    )

    logger.info(
        "SFT training completed."
    )

    return {
        "status": "success",
        "training_type": "sft",
        "model_repo":
        config.model_repo,
        "dataset_repo":
        config.dataset_repo,
        "checkpoint_dir":
        str(SFT_OUTPUT_DIR),
    }


@app.local_entrypoint()
def main():
    """
    Local Modal entrypoint.

    Example:
        modal run
        backend/app/training/modal/modal_sft_runner.py
    """

    result = run_sft_training.remote()

    print(result)
