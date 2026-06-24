# medical-triage-agent-ai-poc/
# backend/app/training/shared/training_utils.py

from __future__ import annotations

import json
import logging
import os
import random
from datetime import datetime
from pathlib import Path
from typing import Any
from typing import Dict
from typing import Optional

import numpy as np
import torch

logger = logging.getLogger(__name__)


class TrainingUtils:
    """
    Shared utilities for training pipelines.

    Supported pipelines:
        - SFT
        - DPO

    Responsibilities:
        - Reproducibility
        - Logging
        - W&B initialization
        - Checkpoint directories
        - Training metadata
        - Metrics persistence

    Non-responsibilities:
        - Model loading
        - Tokenizer loading
        - Dataset loading
        - Clinical evaluation
        - HF Hub upload
    """

    @staticmethod
    def set_seed(seed: int) -> None:
        """
        Configure deterministic training.
        """

        logger.info("Setting random seed: %s", seed)

        random.seed(seed)
        np.random.seed(seed)

        torch.manual_seed(seed)

        if torch.cuda.is_available():
            torch.cuda.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)

        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    @staticmethod
    def setup_logging(
        log_level: str = "INFO",
    ) -> None:
        """
        Configure application logging.
        """

        logging.basicConfig(
            level=getattr(
                logging,
                log_level.upper(),
                logging.INFO,
            ),
            format=(
                "%(asctime)s | "
                "%(levelname)s | "
                "%(name)s | "
                "%(message)s"
            ),
        )

    @staticmethod
    def initialize_wandb(
        config: Dict[str, Any],
    ) -> Optional[Any]:
        """
        Initialize Weights & Biases.

        Returns:
            wandb run object or None.
        """

        wandb_enabled = (
            config.get("wandb", {})
            .get("enabled", False)
        )

        if not wandb_enabled:
            logger.info("W&B disabled")
            return None

        try:
            import wandb

            run = wandb.init(
                project=config["wandb"]["project"],
                entity=config["wandb"].get(
                    "entity",
                ),
                name=config["wandb"].get(
                    "run_name",
                ),
                config=config,
            )

            logger.info("W&B initialized")

            return run

        except Exception as exc:
            logger.exception(
                "Unable to initialize W&B: %s",
                exc,
            )
            return None

    @staticmethod
    def create_directory(
        directory: str | Path,
    ) -> Path:
        """
        Create directory if missing.
        """

        path = Path(directory)

        path.mkdir(
            parents=True,
            exist_ok=True,
        )

        return path

    @staticmethod
    def create_run_directory(
        base_directory: str | Path,
        run_name: Optional[str] = None,
    ) -> Path:
        """
        Create unique run directory.

        Example:
            outputs/sft/20260624_120000/
        """

        timestamp = datetime.utcnow().strftime(
            "%Y%m%d_%H%M%S"
        )

        run_id = (
            f"{timestamp}_{run_name}"
            if run_name
            else timestamp
        )

        run_directory = (
            Path(base_directory) / run_id
        )

        run_directory.mkdir(
            parents=True,
            exist_ok=True,
        )

        logger.info(
            "Run directory created: %s",
            run_directory,
        )

        return run_directory

    @staticmethod
    def save_json(
        data: Dict[str, Any],
        output_file: str | Path,
    ) -> None:
        """
        Save JSON file.
        """

        output_path = Path(output_file)

        with open(
            output_path,
            "w",
            encoding="utf-8",
        ) as file:
            json.dump(
                data,
                file,
                indent=2,
                ensure_ascii=False,
            )

        logger.info(
            "JSON saved: %s",
            output_path,
        )

    @staticmethod
    def save_training_config(
        config: Dict[str, Any],
        output_directory: str | Path,
    ) -> Path:
        """
        Save training configuration.
        """

        output_path = (
            Path(output_directory)
            / "training_config.json"
        )

        TrainingUtils.save_json(
            config,
            output_path,
        )

        return output_path

    @staticmethod
    def save_metrics(
        metrics: Dict[str, Any],
        output_directory: str | Path,
        filename: str = "metrics.json",
    ) -> Path:
        """
        Save training metrics.
        """

        output_path = (
            Path(output_directory)
            / filename
        )

        TrainingUtils.save_json(
            metrics,
            output_path,
        )

        return output_path

    @staticmethod
    def build_training_metadata(
        config: Dict[str, Any],
    ) -> Dict[str, Any]:
        """
        Build metadata used for
        reproducibility and auditing.
        """

        metadata = {
            "timestamp": datetime.utcnow().isoformat(),
            "pythonhashseed": os.environ.get(
                "PYTHONHASHSEED"
            ),
            "torch_version": torch.__version__,
            "cuda_available": torch.cuda.is_available(),
            "cuda_device_count": (
                torch.cuda.device_count()
                if torch.cuda.is_available()
                else 0
            ),
            "seed": config.get(
                "training",
                {},
            ).get(
                "seed",
                None,
            ),
            "base_model": config.get(
                "model",
                {},
            ).get(
                "base_model",
                None,
            ),
        }

        if torch.cuda.is_available():
            metadata["gpu_name"] = (
                torch.cuda.get_device_name(0)
            )

        return metadata

    @staticmethod
    def save_training_metadata(
        metadata: Dict[str, Any],
        output_directory: str | Path,
    ) -> Path:
        """
        Persist metadata.
        """

        output_path = (
            Path(output_directory)
            / "training_metadata.json"
        )

        TrainingUtils.save_json(
            metadata,
            output_path,
        )

        return output_path

    @staticmethod
    def finalize_wandb_run(
        run: Optional[Any],
    ) -> None:
        """
        Properly close W&B session.
        """

        if run is None:
            return

        try:
            run.finish()

            logger.info(
                "W&B run finalized"
            )

        except Exception as exc:
            logger.exception(
                "Error closing W&B run: %s",
                exc,
            )

    @staticmethod
    def get_device_summary() -> Dict[str, Any]:
        """
        Hardware summary used for logs
        and metadata.
        """

        summary = {
            "cuda_available": torch.cuda.is_available(),
            "device_count": (
                torch.cuda.device_count()
                if torch.cuda.is_available()
                else 0
            ),
        }

        if torch.cuda.is_available():
            summary["device_name"] = (
                torch.cuda.get_device_name(0)
            )

        return summary
