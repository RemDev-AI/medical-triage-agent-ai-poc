# medical-triage-agent-ai-poc/backend/app/training/kaggle/kaggle_hf_login.py

"""
Hugging Face Authentication Utilities for Kaggle Notebooks

Features
--------
- Secure Hugging Face authentication
- HF_TOKEN_06 environment variable support
- Kaggle Secrets ("Add-ons > Secrets") support
- Interactive login fallback (Kaggle Notebook / local session)
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

Notes on Kaggle specifics
--------------------------
- Kaggle Notebooks provide a dedicated "Secrets" add-on
  (`kaggle_secrets.UserSecretsClient`) to store credentials without
  exposing them in the notebook source or its committed output. This is
  the recommended way to provide the Hugging Face token on Kaggle,
  since environment variables are not persisted across sessions the way
  Colab's userdata/env can be, and Kaggle's "Batch" (scheduled/committed)
  run mode has no interactive input at all.
- Token resolution order therefore differs slightly from Colab:
    1. Explicit token argument
    2. HF_TOKEN_06 environment variable
    3. Kaggle Secrets (`HF_TOKEN_06` secret, via UserSecretsClient)
    4. Interactive login (only works in an "Interactive" session type;
       will fail in "Batch"/committed runs, which have no stdin)
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

HF_MODELS_REPO_ID = (
    "RemDev-AI/medical-triage-agent-ai-poc-models"
)

HF_DATASETS_REPO_ID = (
    "RemDev-AI/medical-triage-agent-ai-poc-datasets"
)

# Name of the secret to look up via Kaggle Secrets, kept consistent
# with the HF_TOKEN_06 environment variable name used elsewhere.
KAGGLE_HF_SECRET_NAME = "HF_TOKEN_06"


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


def get_hf_token_from_env() -> Optional[str]:
    """
    Return HF token from environment.

    Expected variable:
        HF_TOKEN_06
    """

    token = os.getenv("HF_TOKEN_06")

    if token:
        token = token.strip()

    return token or None


def get_hf_token_from_kaggle_secrets() -> Optional[str]:
    """
    Return HF token from Kaggle Secrets ("Add-ons > Secrets").

    Requires the `kaggle_secrets` package, only available inside a
    Kaggle Notebook runtime. Returns None silently if unavailable
    (e.g. local development, or the secret was not attached to the
    session), so callers can safely fall through to the next
    resolution strategy.
    """

    try:
        from kaggle_secrets import UserSecretsClient

        user_secrets = UserSecretsClient()

        token = user_secrets.get_secret(KAGGLE_HF_SECRET_NAME)

        if token:
            token = token.strip()

        return token or None

    except Exception:
        logger.debug(
            "Kaggle Secrets unavailable or secret '%s' not attached.",
            KAGGLE_HF_SECRET_NAME,
        )
        return None


def get_hf_token() -> Optional[str]:
    """
    Resolve HF token across every supported source.

    Priority:
    1. HF_TOKEN_06 environment variable
    2. Kaggle Secrets (HF_TOKEN_06)
    """

    token = get_hf_token_from_env()

    if token:
        return token

    return get_hf_token_from_kaggle_secrets()


def login_to_huggingface(
    token: Optional[str] = None,
    add_to_git_credential: bool = False,
) -> bool:
    """
    Authenticate against Hugging Face Hub.

    Priority:
    1. Explicit token argument
    2. HF_TOKEN_06 environment variable
    3. Kaggle Secrets (HF_TOKEN_06)
    4. Interactive login (Interactive session type only)
    """

    try:
        resolved_token = token or get_hf_token()

        if resolved_token:
            login(
                token=resolved_token,
                add_to_git_credential=add_to_git_credential,
            )

            logger.info(
                "Successfully authenticated using HF_TOKEN_06."
            )

            return True

        logger.info(
            "No HF_TOKEN_06 found (env or Kaggle Secrets). "
            "Starting interactive login."
        )

        login(
            add_to_git_credential=add_to_git_credential,
        )

        logger.info(
            "Interactive Hugging Face login successful."
        )

        return True

    except Exception:
        logger.exception(
            "Failed to authenticate with Hugging Face. "
            "On Kaggle, verify that the '%s' secret is attached to "
            "this notebook session (Add-ons > Secrets), or that the "
            "run type is 'Interactive' if relying on interactive login.",
            KAGGLE_HF_SECRET_NAME,
        )
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
        logger.exception(
            "Failed validating models repository."
        )
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
        logger.exception(
            "Failed validating dataset repository."
        )
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
            raise RuntimeError(
                "Unable to authenticate to Hugging Face Hub."
            )

    status = validate_hf_access()

    if not status.models_repo_access:
        raise RuntimeError(
            f"Unable to access model repository: "
            f"{HF_MODELS_REPO_ID}"
        )

    if not status.datasets_repo_access:
        raise RuntimeError(
            f"Unable to access dataset repository: "
            f"{HF_DATASETS_REPO_ID}"
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
