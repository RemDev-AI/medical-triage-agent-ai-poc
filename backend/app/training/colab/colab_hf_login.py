# medical-triage-agent-ai-poc/backend/app/training/colab/colab_hf_login.py

"""
Hugging Face Authentication Utilities for Google Colab

Features
--------
- Secure Hugging Face authentication
- HF_TOKEN_06 environment variable support
- Interactive Colab login support
- Repository access validation
- Authentication status checks
- Hugging Face Hub API helpers

Repositories
------------
Models:
    RemDev-AI/medical-triage-agent-ai-poc-models

Datasets:
    RemDev-AI/medical-triage-agent-ai-poc-datasets

Used by:
- backend/app/training/sft/train_sft.py
- backend/app/training/dpo/train_dpo.py
- backend/app/training/evaluation/clinical_eval_runner.py
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass
from typing import Optional

from huggingface_hub import HfApi
from huggingface_hub import login
from huggingface_hub.utils import HfHubHTTPError

logger = logging.getLogger(__name__)

HF_MODELS_REPO_ID = "RemDev-AI/medical-triage-agent-ai-poc-models"

HF_DATASETS_REPO_ID = "RemDev-AI/medical-triage-agent-ai-poc-datasets"


@dataclass
class HuggingFaceAuthStatus:
    """
    Authentication status information.
    """

    authenticated: bool
    username: Optional[str]

    models_repo_access: bool
    datasets_repo_access: bool

    models_repo_id: str
    datasets_repo_id: str


def get_hf_token() -> Optional[str]:
    """
    Return HF token from environment.

    Expected variable:
        HF_TOKEN_06
    """

    token = os.getenv("HF_TOKEN_06")

    if token:
        token = token.strip()

    return token or None


def login_to_huggingface(
    token: Optional[str] = None,
    add_to_git_credential: bool = False,
) -> bool:
    """
    Authenticate against Hugging Face Hub.

    Priority:
    1. Explicit token argument
    2. HF_TOKEN_06 environment variable
    3. Interactive login
    """

    try:
        resolved_token = token or get_hf_token()

        if resolved_token:
            login(
                token=resolved_token,
                add_to_git_credential=add_to_git_credential,
            )

            logger.info("Successfully authenticated using HF_TOKEN_06.")

            return True

        logger.info("No HF_TOKEN_06 found. Starting interactive login.")

        login(
            add_to_git_credential=add_to_git_credential,
        )

        logger.info("Interactive Hugging Face login successful.")

        return True

    except Exception:
        logger.exception("Failed to authenticate with Hugging Face.")
        return False


def get_authenticated_user() -> Optional[str]:
    """
    Return authenticated username.
    """

    try:
        api = HfApi()
        user_info = api.whoami()

        if isinstance(user_info, dict):
            return user_info.get("name")

        return None

    except Exception:
        return None


def is_hf_authenticated() -> bool:
    """
    Check authentication status.
    """

    return get_authenticated_user() is not None


def validate_models_repo_access(
    repo_id: str = HF_MODELS_REPO_ID,
) -> bool:
    """
    Validate access to Models repository.
    """

    try:
        api = HfApi()

        api.repo_info(
            repo_id=repo_id,
            repo_type="model",
        )

        return True

    except HfHubHTTPError:
        return False

    except Exception:
        logger.exception("Failed validating models repository.")
        return False


def validate_datasets_repo_access(
    repo_id: str = HF_DATASETS_REPO_ID,
) -> bool:
    """
    Validate access to Dataset repository.
    """

    try:
        api = HfApi()

        api.repo_info(
            repo_id=repo_id,
            repo_type="dataset",
        )

        return True

    except HfHubHTTPError:
        return False

    except Exception:
        logger.exception("Failed validating dataset repository.")
        return False


def validate_hf_access() -> HuggingFaceAuthStatus:
    """
    Validate authentication and repository access.
    """

    username = get_authenticated_user()

    return HuggingFaceAuthStatus(
        authenticated=username is not None,
        username=username,
        models_repo_access=validate_models_repo_access(),
        datasets_repo_access=validate_datasets_repo_access(),
        models_repo_id=HF_MODELS_REPO_ID,
        datasets_repo_id=HF_DATASETS_REPO_ID,
    )


def ensure_hf_login() -> HuggingFaceAuthStatus:
    """
    Ensure authentication exists and validate access.
    """

    if not is_hf_authenticated():
        success = login_to_huggingface()

        if not success:
            raise RuntimeError("Unable to authenticate to Hugging Face Hub.")

    status = validate_hf_access()

    if not status.models_repo_access:
        raise RuntimeError(
            f"Unable to access model repository: " f"{HF_MODELS_REPO_ID}"
        )

    if not status.datasets_repo_access:
        raise RuntimeError(
            f"Unable to access dataset repository: " f"{HF_DATASETS_REPO_ID}"
        )

    logger.info(
        "Hugging Face authentication and repository "
        "validation completed successfully."
    )

    return status


def get_hf_api() -> HfApi:
    """
    Return authenticated HfApi instance.
    """

    ensure_hf_login()
    return HfApi()


def log_hf_status() -> HuggingFaceAuthStatus:
    """
    Log authentication status.
    """

    status = validate_hf_access()

    logger.info("========== HUGGING FACE ==========")
    logger.info(
        "Authenticated: %s",
        status.authenticated,
    )
    logger.info(
        "Username: %s",
        status.username,
    )
    logger.info(
        "Models Repo: %s",
        status.models_repo_id,
    )
    logger.info(
        "Models Access: %s",
        status.models_repo_access,
    )
    logger.info(
        "Datasets Repo: %s",
        status.datasets_repo_id,
    )
    logger.info(
        "Datasets Access: %s",
        status.datasets_repo_access,
    )
    logger.info("==================================")

    return status


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    try:
        ensure_hf_login()

        status = log_hf_status()

        print()
        print("Authentication Summary")
        print("----------------------")
        print(status)

    except Exception as exc:
        logger.error(
            "Authentication failed: %s",
            exc,
        )
        raise
