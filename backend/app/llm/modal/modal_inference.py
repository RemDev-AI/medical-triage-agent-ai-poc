# medical-triage-agent-ai-poc/backend/app/llm/modal/modal_inference.py

"""
Business inference layer for Modal GPU endpoint.
"""

from __future__ import annotations

import logging
import time

from app.llm.modal.modal_client import ModalClient
from app.llm.modal.modal_schema import (
    ModalGenerationParameters,
    ModalInferenceRequest,
    ModalInferenceResponse,
)

logger = logging.getLogger(__name__)


class ModalInferenceService:
    """
    High-level inference service used by
    the medical triage engine.
    """

    def __init__(
        self,
        modal_client: ModalClient,
    ) -> None:
        self.modal_client = modal_client

    async def generate(
        self,
        system_prompt: str,
        user_prompt: str,
        max_new_tokens: int = 256,
        temperature: float = 0.2,
        top_p: float = 0.9,
        repetition_penalty: float = 1.1,
    ) -> ModalInferenceResponse:
        """
        Execute inference through Modal.
        """

        request = ModalInferenceRequest(
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            generation=ModalGenerationParameters(
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                top_p=top_p,
                repetition_penalty=repetition_penalty,
            ),
        )

        logger.info(
            "Starting Modal inference."
        )

        started_at = time.perf_counter()

        response = await self.modal_client.generate(
            request=request,
        )

        duration_ms = (
            time.perf_counter() - started_at
        ) * 1000

        logger.info(
            (
                "Modal inference completed "
                "in %.2f ms."
            ),
            duration_ms,
        )

        return response

    async def generate_text(
        self,
        system_prompt: str,
        user_prompt: str,
        max_new_tokens: int = 256,
        temperature: float = 0.2,
        top_p: float = 0.9,
        repetition_penalty: float = 1.1,
    ) -> str:
        """
        Generate text only.
        Convenience wrapper used by
        inference modules.
        """

        response = await self.generate(
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_p=top_p,
            repetition_penalty=repetition_penalty,
        )

        return response.generated_text

    async def healthcheck(
        self,
    ) -> bool:
        """
        Verify Modal endpoint availability.
        """

        return await self.modal_client.healthcheck()

    async def warmup(
        self,
    ) -> bool:
        """
        Execute lightweight request to
        warm Modal container.
        """

        try:

            await self.generate_text(
                system_prompt=(
                    "You are a health assistant."
                ),
                user_prompt="ping",
                max_new_tokens=4,
                temperature=0.0,
            )

            logger.info(
                "Modal warmup successful."
            )

            return True

        except Exception:

            logger.exception(
                "Modal warmup failed."
            )

            return False
