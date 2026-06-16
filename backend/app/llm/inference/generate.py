# medical-triage-agent-ai-poc/backend/app/llm/inference/generate.py

"""
Inference generation utilities using Modal GPU.
"""

from __future__ import annotations

import logging
from typing import Dict

from backend.app.llm.modal.modal_inference import (
    ModalInferenceService,
)

logger = logging.getLogger(__name__)


async def generate_response(
    modal_service: ModalInferenceService,
    system_prompt: str,
    user_prompt: str,
    max_new_tokens: int = 256,
    temperature: float = 0.2,
    top_p: float = 0.9,
    repetition_penalty: float = 1.1,
) -> str:
    """
    Generate medical inference response
    through Modal GPU infrastructure.
    """

    logger.info(
        "Generating inference response via Modal."
    )

    response = await modal_service.generate_text(
        system_prompt=system_prompt,
        user_prompt=user_prompt,
        max_new_tokens=max_new_tokens,
        temperature=temperature,
        top_p=top_p,
        repetition_penalty=repetition_penalty,
    )

    logger.info(
        "Inference generation completed."
    )

    return response.strip()


def clean_response(
    response: str,
) -> str:
    """
    Clean malformed outputs returned by LLM.
    """

    response = response.replace(
        "<|im_end|>",
        "",
    )

    response = response.replace(
        "<|endoftext|>",
        "",
    )

    response = response.replace(
        "</s>",
        "",
    )

    response = response.replace(
        "<s>",
        "",
    )

    return response.strip()


def build_generation_metadata(
    latency_seconds: float,
    model_name: str,
) -> Dict:
    """
    Build inference metadata.
    """

    return {
        "latency_seconds": round(
            latency_seconds,
            2,
        ),
        "model_name": model_name,
    }
