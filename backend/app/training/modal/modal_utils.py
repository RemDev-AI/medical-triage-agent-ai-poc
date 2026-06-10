# medical-triage-agent-ai-poc/backend/app/training/modal/modal_utils.py

from __future__ import annotations

import json
import logging
import os
from datetime import datetime
from pathlib import Path
from typing import Any
from typing import Dict
from typing import Optional

from datasets import load_dataset
from huggingface_hub import HfApi
from huggingface_hub import login
from huggingface_hub import snapshot_download

from app.training.modal.modal_config import (
    CHECKPOINT_DIR,
    DATASET_DIR,
    HF_DATASET_REPO,
    HF_MODEL_REPO,
    training_volume,
)

logger = logging.getLogger(__name__)


# ============================================================
# Environment helpers
# ============================================================


def get_env(
    key: str,
    default: Optional[str] = None,
) -> str:
    """
    Read environment variable.

    Raises:
        RuntimeError if variable is missing.
    """

    value = os.getenv(key, default)

    if value is None:
        raise RuntimeError(
            f"Missing required environment variable: {key}"
        )

    return value


def get_hf_token() -> str:
    """
    Hugging Face access token.
    """

    return get_env("HF_TOKEN")


def get_wandb_api_key() -> str:
    """
    Weights & Biases API key.
    """

    return get_env("WANDB_API_KEY")


def get_mlflow_tracking_uri() -> str:
    """
    MLflow tracking endpoint.
    """

    return get_env("MLFLOW_TRACKING_URI")


# ============================================================
# Hugging Face Authentication
# ============================================================


def authenticate_huggingface() -> None:
    """
    Login to Hugging Face Hub.
    """

    logger.info(
        "Authenticating against Hugging Face Hub..."
    )

    login(
        token=get_hf_token(),
        add_to_git_credential=False,
    )

    logger.info(
        "Hugging Face authentication successful."
    )


def get_hf_api() -> HfApi:
    """
    Return authenticated HF API client.
    """

    authenticate_huggingface()

    return HfApi(
        token=get_hf_token(),
    )


# ============================================================
# Directories
# ============================================================


def ensure_directory(path: str | Path) -> Path:
    """
    Create directory if needed.
    """

    path = Path(path)

    path.mkdir(
        parents=True,
        exist_ok=True,
    )

    return path


def prepare_training_directories() -> Dict[str, Path]:
    """
    Create all required training directories.
    """

    dataset_dir = ensure_directory(
        DATASET_DIR
    )

    checkpoint_dir = ensure_directory(
        CHECKPOINT_DIR
    )

    logs_dir = ensure_directory(
        Path(CHECKPOINT_DIR) / "logs"
    )

    metadata_dir = ensure_directory(
        Path(CHECKPOINT_DIR) / "metadata"
    )

    return {
        "dataset_dir": dataset_dir,
        "checkpoint_dir": checkpoint_dir,
        "logs_dir": logs_dir,
        "metadata_dir": metadata_dir,
    }


# ============================================================
# Dataset management
# ============================================================


def download_dataset(
    dataset_name: str,
    split: Optional[str] = None,
):
    """
    Download HF dataset.

    Example:
        medical-triage-agent-ai-poc-datasets
    """

    authenticate_huggingface()

    logger.info(
        "Downloading dataset %s",
        dataset_name,
    )

    dataset = load_dataset(
        dataset_name,
        split=split,
        token=get_hf_token(),
    )

    logger.info(
        "Dataset downloaded successfully."
    )

    return dataset


def download_project_dataset(
    split: Optional[str] = None,
):
    """
    Download project dataset.
    """

    return download_dataset(
        HF_DATASET_REPO,
        split=split,
    )


# ============================================================
# Model repository helpers
# ============================================================


def ensure_model_repository_exists() -> None:
    """
    Ensure HF Models repository exists.
    """

    api = get_hf_api()

    try:
        api.repo_info(
            repo_id=HF_MODEL_REPO,
            repo_type="model",
        )

        logger.info(
            "HF model repository exists."
        )

    except Exception:

        logger.warning(
            "Repository not found. Creating..."
        )

        api.create_repo(
            repo_id=HF_MODEL_REPO,
            repo_type="model",
            private=False,
            exist_ok=True,
        )

        logger.info(
            "HF model repository created."
        )


# ============================================================
# Checkpoint upload
# ============================================================


def upload_checkpoint(
    checkpoint_path: str | Path,
    commit_message: str,
) -> None:
    """
    Upload training checkpoint to HF Models.
    """

    checkpoint_path = Path(checkpoint_path)

    ensure_model_repository_exists()

    api = get_hf_api()

    logger.info(
        "Uploading checkpoint %s",
        checkpoint_path,
    )

    api.upload_folder(
        folder_path=str(checkpoint_path),
        repo_id=HF_MODEL_REPO,
        repo_type="model",
        commit_message=commit_message,
    )

    logger.info(
        "Checkpoint uploaded successfully."
    )


def upload_final_model(
    model_path: str | Path,
    stage: str,
) -> None:
    """
    Upload final SFT/DPO model.
    """

    timestamp = datetime.utcnow().strftime(
        "%Y%m%d_%H%M%S"
    )

    commit_message = (
        f"{stage.upper()} model upload "
        f"{timestamp}"
    )

    upload_checkpoint(
        checkpoint_path=model_path,
        commit_message=commit_message,
    )


# ============================================================
# Snapshot helpers
# ============================================================


def download_latest_model_snapshot(
    local_dir: str | Path,
) -> Path:
    """
    Download latest model snapshot.
    """

    authenticate_huggingface()

    local_dir = Path(local_dir)

    snapshot_download(
        repo_id=HF_MODEL_REPO,
        repo_type="model",
        local_dir=str(local_dir),
        token=get_hf_token(),
    )

    return local_dir


# ============================================================
# Metadata helpers
# ============================================================


def build_training_metadata(
    training_type: str,
    base_model: str,
    dataset_name: str,
    extra: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """
    Build metadata payload.
    """

    payload = {
        "training_type": training_type,
        "base_model": base_model,
        "dataset_name": dataset_name,
        "created_at": datetime.utcnow().isoformat(),
    }

    if extra:
        payload.update(extra)

    return payload


def save_training_metadata(
    metadata: Dict[str, Any],
    output_file: str | Path,
) -> None:
    """
    Persist metadata JSON.
    """

    output_file = Path(output_file)

    ensure_directory(
        output_file.parent
    )

    with open(
        output_file,
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            metadata,
            file,
            indent=2,
            ensure_ascii=False,
        )

    logger.info(
        "Training metadata saved: %s",
        output_file,
    )


# ============================================================
# Modal volume helpers
# ============================================================


def commit_volume() -> None:
    """
    Persist volume changes.
    """

    logger.info(
        "Persisting Modal volume..."
    )

    training_volume.commit()

    logger.info(
        "Modal volume committed."
    )


def reload_volume() -> None:
    """
    Refresh volume state.
    """

    logger.info(
        "Reloading Modal volume..."
    )

    training_volume.reload()

    logger.info(
        "Modal volume reloaded."
    )
