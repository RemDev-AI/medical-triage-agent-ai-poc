# medical-triage-agent-ai-poc/backend/app/training/colab/colab_checkpoint_sync.py

"""
Google Colab Checkpoint Synchronization Utilities

Features
--------
- Local checkpoint management
- Hugging Face Models synchronization
- Automatic checkpoint discovery
- Resume-from-checkpoint support (restauration depuis Hugging Face)

Used by:
- backend/app/training/sft/train_sft.py
- backend/app/training/dpo/train_dpo.py

Architecture
------------
Hugging Face Models Repository:
    RemDev-AI/medical-triage-agent-ai-poc-models

NETTOYAGE (2026-07-07) — méthodes supprimées car jamais appelées par
train_sft.py ni train_dpo.py, seuls consommateurs réels de ce module :
    - sync_latest_checkpoint() / sync_latest_checkpoint_to_huggingface()
      -> les deux scripts appellent sync_all_checkpoints_to_huggingface(),
         qui boucle déjà sur chaque checkpoint via
         sync_checkpoint_to_huggingface(). Le wrapper "latest" était un
         doublon partiel, non branché sur le pipeline réel.
    - get_resume_checkpoint() / auto_restore_checkpoint()
      -> les deux scripts appellent directement
         restore_latest_checkpoint_from_huggingface() (pas
         auto_restore_checkpoint), et ne lisent jamais
         get_resume_checkpoint().
    - get_status()
      -> ne reflète que l'état LOCAL (get_checkpoints() sur
         local_checkpoint_dir), jamais l'état sur le Hub. Ni train_sft.py
         ni train_dpo.py ne l'utilisent. Source de confusion : à un
         instant T hors training, local_checkpoint_dir est presque
         toujours vide (tout est déjà poussé + nettoyé sur HF), ce qui
         rend cette méthode trompeuse pour un diagnostic manuel.
    - bloc `if __name__ == "__main__":`
      -> code de test mort, jamais exécuté par le pipeline (runpy cible
         train_sft/train_dpo/clinical_eval_runner, jamais ce module).

Si un besoin de diagnostic manuel (vérifier ce qui existe sur le Hub)
réapparaît, utiliser directement
`restore_latest_checkpoint_from_huggingface()` ou `get_checkpoints()`
depuis un notebook, plutôt que de réintroduire ces wrappers.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

from huggingface_hub import HfApi
from huggingface_hub import snapshot_download

logger = logging.getLogger(__name__)

HF_MODELS_REPO_ID = "RemDev-AI/medical-triage-agent-ai-poc-models"


class ColabCheckpointSync:
    """
    Manage training checkpoints across:

    - Local filesystem
    - Hugging Face Models repository
    """

    def __init__(
        self,
        local_checkpoint_dir: str,
        training_type: str,
        hf_repo_id: str = HF_MODELS_REPO_ID,
        cleanup_after_upload: bool = True,
        revision: str = "main",
    ) -> None:

        self.local_checkpoint_dir = Path(local_checkpoint_dir)

        self.training_type = training_type.lower()

        if self.training_type not in {"sft", "dpo"}:
            raise ValueError("training_type must be 'sft' or 'dpo'.")

        self.hf_repo_id = hf_repo_id
        self.cleanup_after_upload = cleanup_after_upload
        # Pin the Hub revision (commit SHA, tag, or branch) used when
        # downloading checkpoints, to avoid an unexpected concurrent push
        # changing content mid-restore (Bandit B615). "main" preserves the
        # existing "always get the latest" behavior.
        self.revision = revision

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
            if checkpoint.is_dir() and checkpoint.name.startswith("checkpoint-")
        ]

        return sorted(
            checkpoints,
            key=lambda x: int(x.name.replace("checkpoint-", "")),
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
        Check if checkpoints exist locally.
        """

        return self.get_latest_checkpoint() is not None

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

        return f"checkpoint-" f"{self.training_type}-" f"{step}"

    def cleanup_checkpoint(
        self,
        checkpoint_path: Path,
    ) -> bool:
        """
        Delete local checkpoint after successful upload.
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

            logger.exception("Checkpoint cleanup failed.")

            return False

    def sync_checkpoint_to_huggingface(
        self,
        checkpoint_path: Path,
    ) -> bool:
        """
        Upload a checkpoint to the Hugging Face
        Models repository.
        """

        remote_name = self._build_remote_checkpoint_name(checkpoint_path)

        remote_dir = "checkpoints/" f"{self.training_type}/" f"{remote_name}"

        try:

            api = HfApi()

            api.upload_folder(
                folder_path=str(checkpoint_path),
                repo_id=self.hf_repo_id,
                repo_type="model",
                path_in_repo=remote_dir,
                commit_message=(f"Upload {remote_name}"),
            )

            logger.info(
                "Uploaded checkpoint %s",
                remote_name,
            )

            # FIX HUB-4 — ne pas faire confiance à la seule absence
            # d'exception de upload_folder(). On relit la liste des
            # fichiers réellement présents sur le Hub sous remote_dir/
            # et on la compare aux fichiers locaux attendus AVANT tout
            # nettoyage local (cleanup_checkpoint supprime le checkpoint
            # de façon irréversible).
            local_files = {
                str(path.relative_to(checkpoint_path)).replace("\\", "/")
                for path in checkpoint_path.rglob("*")
                if path.is_file()
            }

            remote_prefix = f"{remote_dir}/"
            remote_files = {
                file[len(remote_prefix) :]
                for file in api.list_repo_files(
                    repo_id=self.hf_repo_id,
                    repo_type="model",
                )
                if file.startswith(remote_prefix)
            }

            missing_files = local_files - remote_files

            if missing_files:
                logger.error(
                    "Vérification post-upload échouée pour %s : %d "
                    "fichier(s) manquant(s) sur le Hub : %s. "
                    "Nettoyage local ANNULÉ par précaution.",
                    remote_name,
                    len(missing_files),
                    sorted(missing_files),
                )
                return False

            logger.info(
                "Vérification post-upload OK pour %s (%d fichier(s)).",
                remote_name,
                len(local_files),
            )

            if self.cleanup_after_upload:

                self.cleanup_checkpoint(checkpoint_path)

            return True

        except Exception:

            logger.exception("Checkpoint upload failed.")

            return False

    def sync_all_checkpoints_to_huggingface(
        self,
    ) -> bool:
        """
        Upload every checkpoint found locally.
        """

        checkpoints = self.get_checkpoints()

        if not checkpoints:
            logger.warning("No checkpoints found.")
            return False

        success = True

        for checkpoint in checkpoints:
            uploaded = self.sync_checkpoint_to_huggingface(checkpoint)

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

            prefix = f"checkpoints/{self.training_type}/"

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
                key=lambda name: int(name.split("-")[-1]),
            )

            logger.info(
                "Latest checkpoint detected: %s",
                latest_checkpoint,
            )

            local_path = snapshot_download(
                repo_id=self.hf_repo_id,
                repo_type="model",
                revision=self.revision,
                allow_patterns=[(f"{prefix}" f"{latest_checkpoint}/*")],
                local_dir=str(self.local_checkpoint_dir),
                local_dir_use_symlinks=False,
            )

            checkpoint_path = Path(local_path) / prefix / latest_checkpoint

            # FIX HUB-6 — snapshot_download(local_dir=...) crée
            # systématiquement un sous-dossier "<local_dir>/.cache/huggingface/"  # noqa : E501
            # (fichiers *.metadata de suivi du téléchargement incrémental).
            # Comme local_dir == self.local_checkpoint_dir == output_dir
            # (cf. create_default_checkpoint_sync), ce cache technique se
            # retrouve mélangé au répertoire où train_sft.py sauvegarde
            # ensuite le modèle final (trainer.save_model()). Sans
            # nettoyage, ces fichiers de cache sont ensuite scannés par la
            # vérification post-upload de train_sft.py (rglob(output_dir)),
            # qui les signale à tort comme "manquants sur le Hub" — alors
            # qu'ils n'ont jamais eu vocation à y être. Ce cache n'a plus
            # aucune utilité une fois le checkpoint restauré : on le
            # supprime immédiatement (best-effort, ne doit jamais faire
            # échouer une restauration par ailleurs réussie).
            cache_dir = self.local_checkpoint_dir / ".cache"
            if cache_dir.exists():
                import shutil

                shutil.rmtree(cache_dir, ignore_errors=True)
                logger.info(
                    "Cache technique huggingface_hub (%s) nettoyé après "
                    "restauration du checkpoint.",
                    cache_dir,
                )

            logger.info(
                "Checkpoint restored into %s",
                checkpoint_path,
            )

            return str(checkpoint_path)

        except Exception:

            logger.exception("Unable to restore checkpoint from Hugging Face.")

            return None


def create_default_checkpoint_sync(
    output_dir: str,
    training_type: str,
) -> ColabCheckpointSync:
    """
    Factory helper.
    """

    return ColabCheckpointSync(
        local_checkpoint_dir=output_dir,
        training_type=training_type,
        hf_repo_id=HF_MODELS_REPO_ID,
    )
