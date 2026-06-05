# medical-triage-agent-ai-poc/backend/app/deployment/modal/modal_image.py

"""
modal_image.py

Construction de l'image Modal utilisée pour :

- vLLM
- Qwen3
- LoRA adapters
- FastAPI endpoints
- Inference Engine

Compatible :

- Modal
- Hugging Face Hub
- Transformers
- PEFT
- vLLM

Objectifs :

- reproductibilité MLOps ;
- image unique pour tous les endpoints ;
- déploiement simplifié ;
- compatibilité GPU Modal.
"""

from __future__ import annotations

import modal


# =========================================================
# CONFIGURATION
# =========================================================

PYTHON_VERSION = "3.11"


# =========================================================
# DEPENDENCIES
# =========================================================

PIP_PACKAGES = [
    # Core AI
    "torch>=2.7.0",
    "transformers>=4.53.0",
    "accelerate>=1.8.0",
    "peft>=0.16.0",

    # Inference
    "vllm>=0.10.0",

    # Hugging Face
    "huggingface-hub>=0.34.0",
    "datasets>=4.0.0",

    # API
    "fastapi>=0.116.0",
    "uvicorn[standard]>=0.35.0",

    # Validation
    "pydantic>=2.11.0",

    # Monitoring
    "mlflow>=3.1.0",
    "wandb>=0.21.0",

    # Medical NLP
    "presidio-analyzer>=2.2.358",
    "presidio-anonymizer>=2.2.358",

    # Utilities
    "numpy>=2.3.0",
    "pandas>=2.3.0",
    "orjson>=3.11.0",
    "python-dotenv>=1.1.0",
    "loguru>=0.7.3",
    "pyyaml>=6.0.2",
    "requests>=2.32.0",
]

# =========================================================
# IMAGE BUILD
# =========================================================

modal_image = (
    modal.Image.debian_slim(
        python_version=PYTHON_VERSION
    )
    .apt_install(
        [
            "git",
            "curl",
            "build-essential",
            "gcc",
            "g++",
        ]
    )
    .pip_install(*PIP_PACKAGES)
    .run_commands(
        [
            "python -m spacy download fr_core_news_md"
        ]
    )
)


# =========================================================
# HF ENVIRONMENT
# =========================================================

modal_image = modal_image.env(
    {
        "HF_HUB_ENABLE_HF_TRANSFER": "1",
        "TOKENIZERS_PARALLELISM": "false",
        "PYTHONUNBUFFERED": "1",
        "TRANSFORMERS_NO_ADVISORY_WARNINGS": "1",
    }
)


# =========================================================
# IMAGE ACCESSOR
# =========================================================

def get_modal_image() -> modal.Image:
    """
    Retourne l'image Modal utilisée
    dans tout le projet.
    """

    return modal_image


# =========================================================
# IMAGE DESCRIPTION
# =========================================================

def image_summary() -> dict:
    """
    Informations utiles pour audit.
    """

    return {
        "python_version": PYTHON_VERSION,
        "package_count": len(PIP_PACKAGES),
        "spacy_model": "fr_core_news_md",
        "vllm_enabled": True,
        "huggingface_enabled": True,
        "medical_nlp_enabled": True,
    }


# =========================================================
# MAIN
# =========================================================

if __name__ == "__main__":

    print(
        "Modal image configuration loaded."
    )

    print(
        image_summary()
    )
