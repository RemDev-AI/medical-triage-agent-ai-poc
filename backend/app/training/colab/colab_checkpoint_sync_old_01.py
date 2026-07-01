# medical-triage-agent-ai-poc/backend/app/training/colab/colab_checkpoint_sync.py
# sync_to_drive()
"""
Google Colab Checkpoint Synchronization Utilities

Features
--------
- Local checkpoint management
- Google Drive synchronization
- Hugging Face Models synchronization
- Automatic checkpoint discovery
- Resume-from-checkpoint support
- Automatic checkpoint uploads during training

Used by:
- backend/app/training/sft/train_sft.py
- backend/app/training/dpo/train_dpo.py
- backend/app/training/evaluation/clinical_eval_runner.py

Architecture
------------
Hugging Face Models Repository:
    RemDev-AI/medical-triage-agent-ai-poc-models
"""

from __future__ import annotations

import logging
# import shutil
import tempfile
from pathlib import Path
from typing import Optional

from huggingface_hub import HfApi

logger = logging.getLogger(__name__)

HF_MODELS_REPO_ID = (
    "RemDev-AI/medical-triage-agent-ai-poc-models"
)


class ColabCheckpointSync:
    """
    Manage training checkpoints across:

    - Local filesystem
    - Google Drive
    - Hugging Face Models repository
    """

    def __init__(
        self,
        local_checkpoint_dir: str,
        training_type: str,
        hf_repo_id: str = HF_MODELS_REPO_ID,
        cleanup_after_upload: bool = True,
    ) -> None:

        self.local_checkpoint_dir = Path(local_checkpoint_dir)

        self.training_type = training_type.lower()

        if self.training_type not in {"sft", "dpo"}:
            raise ValueError(
                "training_type must be 'sft' or 'dpo'."
            )

        self.hf_repo_id = hf_repo_id
        self.cleanup_after_upload = cleanup_after_upload

        self.local_checkpoint_dir.mkdir(
            parents=True,
            exist_ok=True,
        )

    # ==========================================================
    # Local checkpoint management
    # ==========================================================

    def get_checkpoints(self) -> list[Path]:
        """
        Return all checkpoint directories sorted by step.
        """

        if not self.local_checkpoint_dir.exists():
            return []

        checkpoints = [
            checkpoint
            for checkpoint in self.local_checkpoint_dir.iterdir()
            if checkpoint.is_dir()
            and checkpoint.name.startswith("checkpoint-")
        ]

        return sorted(
            checkpoints,
            key=lambda x: int(
                x.name.replace("checkpoint-", "")
            ),
        )

    def get_latest_checkpoint(self) -> Optional[Path]:
        """
        Return latest checkpoint directory.
        """

        checkpoints = self.get_checkpoints()

        if not checkpoints:
            return None

        return checkpoints[-1]

    def has_checkpoint(self) -> bool:
        """
        Check if checkpoints exist.
        """

        return self.get_latest_checkpoint() is not None

    def get_resume_checkpoint(self) -> Optional[str]:
        """
        Return checkpoint path compatible with
        Hugging Face Trainer.resume_from_checkpoint.
        """

        checkpoint = self.get_latest_checkpoint()

        if checkpoint is None:
            return None

        return str(checkpoint)

    # ==========================================================
    # Google Drive synchronization
    # ==========================================================

    def sync_to_drive(self) -> bool:
        """
        Synchronize all checkpoints to Google Drive.
        """

        if self.drive_checkpoint_dir is None:
            logger.warning(
                "Google Drive directory not configured."
            )
            return False

        try:
            self.drive_checkpoint_dir.mkdir(
                parents=True,
                exist_ok=True,
            )

            destination = (
                self.drive_checkpoint_dir
                / self.local_checkpoint_dir.name
            )

            if destination.exists():
                shutil.rmtree(destination)

            shutil.copytree(
                self.local_checkpoint_dir,
                destination,
            )

            logger.info(
                "Checkpoints synchronized to Google Drive: %s",
                destination,
            )

            return True

        except Exception:
            logger.exception(
                "Google Drive synchronization failed."
            )
            return False

    def restore_from_drive(self) -> bool:
        """
        Restore checkpoints from Google Drive.
        """

        if self.drive_checkpoint_dir is None:
            return False

        source = (
            self.drive_checkpoint_dir
            / self.local_checkpoint_dir.name
        )

        if not source.exists():
            logger.warning(
                "No checkpoints found in Google Drive."
            )
            return False

        try:
            if self.local_checkpoint_dir.exists():
                shutil.rmtree(
                    self.local_checkpoint_dir
                )

            shutil.copytree(
                source,
                self.local_checkpoint_dir,
            )

            logger.info(
                "Checkpoints restored from Google Drive."
            )

            return True

        except Exception:
            logger.exception(
                "Failed restoring checkpoints from Drive."
            )
            return False

    # ==========================================================
    # Hugging Face synchronization
    # ==========================================================

    def sync_checkpoint_to_huggingface(
        self,
        checkpoint_path: Path,
    ) -> bool:
        """
        Upload a single checkpoint to HF Models repo.

        Example destination:

        checkpoints/
            checkpoint-100/
            checkpoint-200/
            checkpoint-300/
        """

        try:
            api = HfApi()

            api.upload_folder(
                folder_path=str(checkpoint_path),
                repo_id=self.hf_repo_id,
                repo_type="model",
                path_in_repo=(
                    f"checkpoints/{checkpoint_path.name}"
                ),
                commit_message=(
                    f"Upload {checkpoint_path.name}"
                ),
            )

            logger.info(
                "Checkpoint uploaded to HF Hub: %s",
                checkpoint_path.name,
            )

            return True

        except Exception:
            logger.exception(
                "HF checkpoint upload failed."
            )
            return False

    def sync_latest_checkpoint_to_huggingface(
        self,
    ) -> bool:
        """
        Upload latest checkpoint only.
        """

        latest_checkpoint = (
            self.get_latest_checkpoint()
        )

        if latest_checkpoint is None:
            logger.warning(
                "No checkpoint available."
            )
            return False

        return self.sync_checkpoint_to_huggingface(
            latest_checkpoint
        )

    def sync_all_checkpoints_to_huggingface(
        self,
    ) -> bool:
        """
        Upload every checkpoint found locally.
        """

        checkpoints = self.get_checkpoints()

        if not checkpoints:
            logger.warning(
                "No checkpoints found."
            )
            return False

        success = True

        for checkpoint in checkpoints:
            uploaded = (
                self.sync_checkpoint_to_huggingface(
                    checkpoint
                )
            )

            if not uploaded:
                success = False

        return success

    # ==========================================================
    # Combined synchronization
    # ==========================================================

    def sync_latest_checkpoint(
        self,
        upload_to_drive: bool = True,
        upload_to_huggingface: bool = True,
    ) -> dict:
        """
        Synchronize latest checkpoint everywhere.
        """

        results = {
            "drive": False,
            "huggingface": False,
        }

        if upload_to_drive:
            results["drive"] = self.sync_to_drive()

        if upload_to_huggingface:
            results["huggingface"] = (
                self.sync_latest_checkpoint_to_huggingface()
            )

        return results

    # ==========================================================
    # Status
    # ==========================================================

    def get_status(self) -> dict:
        """
        Return synchronization status.
        """

        latest_checkpoint = (
            self.get_latest_checkpoint()
        )

        return {
            "hf_repo_id": self.hf_repo_id,
            "local_checkpoint_dir": str(
                self.local_checkpoint_dir
            ),
            "drive_checkpoint_dir": (
                str(self.drive_checkpoint_dir)
                if self.drive_checkpoint_dir
                else None
            ),
            "has_checkpoint": (
                latest_checkpoint is not None
            ),
            "latest_checkpoint": (
                str(latest_checkpoint)
                if latest_checkpoint
                else None
            ),
            "checkpoint_count": len(
                self.get_checkpoints()
            ),
        }


def create_default_checkpoint_sync(
    output_dir: str,
    drive_dir: Optional[str] = None,
) -> ColabCheckpointSync:
    """
    Factory helper aligned with project architecture.
    """

    return ColabCheckpointSync(
        local_checkpoint_dir=output_dir,
        drive_checkpoint_dir=drive_dir,
        hf_repo_id=HF_MODELS_REPO_ID,
    )


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    sync = ColabCheckpointSync(
        local_checkpoint_dir="./outputs",
        drive_checkpoint_dir=(
            "/content/drive/MyDrive/"
            "medical-triage-agent-ai-poc/checkpoints"
        ),
    )

    print(sync.get_status())
