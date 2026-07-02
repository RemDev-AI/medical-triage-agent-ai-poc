# medical-triage-agent-ai-poc/backend/app/training/kaggle/kaggle_checkpoint_sync.py

"""
Kaggle Notebooks Checkpoint Synchronization Utilities

Features
--------
- Local checkpoint management
- Kaggle working directory persistence awareness
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

Notes on Kaggle specifics
--------------------------
- Kaggle Notebooks do not provide a persistent mount equivalent to
  Google Drive. The working directory (`/kaggle/working`) is only
  preserved for the duration of the session and is capped in size
  (commonly 20 GB), and outputs are only kept permanently if the
  notebook is committed/saved as a Version.
- Kaggle sessions also have a hard runtime limit (commonly 9h/12h
  depending on the accelerator), which makes frequent Hugging Face
  checkpoint uploads even more important than on Colab, since a
  session can be pre-empted or time out without warning.
- Because of this, `cleanup_after_upload` defaults to True to avoid
  filling up the limited `/kaggle/working` quota with old checkpoints
  once they are safely persisted to the Hugging Face Hub.
"""

from __future__ import annotations

import logging
# import shutil
import tempfile  # noqa : F401
from pathlib import Path
from typing import Optional

from huggingface_hub import HfApi
from huggingface_hub import snapshot_download

logger = logging.getLogger(__name__)

HF_MODELS_REPO_ID = (
    "RemDev-AI/medical-triage-agent-ai-poc-models"
)

# Default Kaggle-friendly local checkpoint location.
# `/kaggle/working` is the only writable, session-scoped directory
# whose content Kaggle will offer to persist when a notebook Version
# is committed.
KAGGLE_DEFAULT_CHECKPOINT_DIR = "/kaggle/working/outputs"


class KaggleCheckpointSync:
    """
    Manage training checkpoints across:

    - Local filesystem (Kaggle `/kaggle/working` session storage)
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
    # Hugging Face synchronization
    # ==========================================================

    def _build_remote_checkpoint_name(
        self,
        checkpoint_path: Path,
    ) -> str:
        """
        Convert checkpoint-500

        into

        checkpoint-sft-500

        or

        checkpoint-dpo-500
        """

        step = checkpoint_path.name.replace(
            "checkpoint-",
            "",
        )

        return (
            f"checkpoint-"
            f"{self.training_type}-"
            f"{step}"
        )

    def cleanup_checkpoint(
        self,
        checkpoint_path: Path,
    ) -> bool:
        """
        Delete local checkpoint after successful upload.

        On Kaggle this is particularly important because
        `/kaggle/working` has a limited quota (commonly 20 GB),
        shared across all outputs of the session.
        """

        try:

            if checkpoint_path.exists():

                import shutil

                shutil.rmtree(checkpoint_path)

                logger.info(
                    "Deleted local checkpoint %s",
                    checkpoint_path,
                )

            return True

        except Exception:

            logger.exception(
                "Checkpoint cleanup failed."
            )

            return False

    def sync_checkpoint_to_huggingface(
        self,
        checkpoint_path: Path,
    ) -> bool:
        """
        Upload a checkpoint to the Hugging Face
        Models repository.
        """

        remote_name = (
            self._build_remote_checkpoint_name(
                checkpoint_path
            )
        )

        try:

            api = HfApi()

            api.upload_folder(

                folder_path=str(checkpoint_path),

                repo_id=self.hf_repo_id,

                repo_type="model",

                path_in_repo=(
                    "checkpoints/"
                    f"{self.training_type}/"
                    f"{remote_name}"
                ),

                commit_message=(
                    f"Upload {remote_name}"
                ),
            )

            logger.info(
                "Uploaded checkpoint %s",
                remote_name,
            )

            if self.cleanup_after_upload:

                self.cleanup_checkpoint(
                    checkpoint_path
                )

            return True

        except Exception:

            logger.exception(
                "Checkpoint upload failed."
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
    # Hugging Face restoration
    # ==========================================================

    def restore_latest_checkpoint_from_huggingface(
        self,
    ) -> Optional[str]:
        """
        Download the latest checkpoint from the Hugging Face
        Models repository and restore it locally.

        Repository layout:

        checkpoints/
            sft/
                checkpoint-sft-500/
                checkpoint-sft-1000/

            dpo/
                checkpoint-dpo-500/
                checkpoint-dpo-1000/

        Returns
        -------
        Optional[str]
            Local checkpoint directory compatible with
            Trainer.resume_from_checkpoint.
        """

        try:

            api = HfApi()

            files = api.list_repo_files(
                repo_id=self.hf_repo_id,
                repo_type="model",
            )

            prefix = (
                f"checkpoints/{self.training_type}/"
            )

            checkpoint_dirs = set()

            for file in files:

                if not file.startswith(prefix):
                    continue

                parts = file.split("/")

                if len(parts) >= 3:
                    checkpoint_dirs.add(parts[2])

            if not checkpoint_dirs:

                logger.warning(
                    "No %s checkpoint found on Hugging Face.",
                    self.training_type,
                )

                return None

            latest_checkpoint = max(

                checkpoint_dirs,

                key=lambda name: int(
                    name.split("-")[-1]
                ),

            )

            logger.info(
                "Latest checkpoint detected: %s",
                latest_checkpoint,
            )

            local_path = snapshot_download(

                repo_id=self.hf_repo_id,

                repo_type="model",

                allow_patterns=[
                    (
                        f"{prefix}"
                        f"{latest_checkpoint}/*"
                    )
                ],

                local_dir=str(
                    self.local_checkpoint_dir
                ),

                local_dir_use_symlinks=False,

            )

            checkpoint_path = (
                Path(local_path)
                / prefix
                / latest_checkpoint
            )

            logger.info(
                "Checkpoint restored into %s",
                checkpoint_path,
            )

            return str(checkpoint_path)

        except Exception:

            logger.exception(
                "Unable to restore checkpoint from Hugging Face."
            )

            return None

    def auto_restore_checkpoint(
        self,
    ) -> Optional[str]:
        """
        Return a checkpoint usable by the Trainer.

        Priority:

        1. Local checkpoint

        2. Hugging Face checkpoint

        3. None
        """

        local_checkpoint = self.get_resume_checkpoint()

        if local_checkpoint is not None:

            logger.info(
                "Using local checkpoint."
            )

            return local_checkpoint

        logger.info(
            "No local checkpoint found. Restoring from Hugging Face..."
        )

        return self.restore_latest_checkpoint_from_huggingface()

    # ==========================================================
    # Combined synchronization
    # ==========================================================

    def sync_latest_checkpoint(
        self,
    ) -> dict:
        """
        Upload latest checkpoint to HF Hub.
        """

        return {

            "huggingface":
            self.sync_latest_checkpoint_to_huggingface()

        }

    # ==========================================================
    # Status
    # ==========================================================

    def get_status(
        self,
    ) -> dict:
        """
        Return synchronization status.
        """

        latest = self.get_latest_checkpoint()

        return {

            "training_type":
            self.training_type,

            "hf_repo_id":
            self.hf_repo_id,

            "local_checkpoint_dir":
            str(self.local_checkpoint_dir),

            "cleanup_after_upload":
            self.cleanup_after_upload,

            "has_checkpoint":
            latest is not None,

            "latest_checkpoint":
            (
                str(latest)
                if latest
                else None
            ),

            "checkpoint_count":
            len(
                self.get_checkpoints()
            ),
        }


def create_default_checkpoint_sync(
    output_dir: str = KAGGLE_DEFAULT_CHECKPOINT_DIR,
    training_type: str = "sft",
) -> KaggleCheckpointSync:
    """
    Factory helper.
    """

    return KaggleCheckpointSync(

        local_checkpoint_dir=output_dir,

        training_type=training_type,

        hf_repo_id=HF_MODELS_REPO_ID,

    )


if __name__ == "__main__":

    logging.basicConfig(
        level=logging.INFO
    )

    sync = KaggleCheckpointSync(

        local_checkpoint_dir=KAGGLE_DEFAULT_CHECKPOINT_DIR,

        training_type="sft",

    )

    print(sync.get_status())
