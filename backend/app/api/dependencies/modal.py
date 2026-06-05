# medical-triage-agent-ai-poc/backend/app/api/dependencies/modal.py

from __future__ import annotations

import os
from typing import Any

import httpx


class ModalInferenceClient:
    """
    Client HTTP vers l'infrastructure Modal GPU.

    Responsabilités :
    - envoyer les prompts
    - gérer les timeouts
    - centraliser l'authentification
    - uniformiser les erreurs
    """

    def __init__(self) -> None:

        self.base_url = os.getenv(
            "MODAL_ENDPOINT_URL",
            "",
        )

        self.api_key = os.getenv(
            "MODAL_API_KEY",
            "",
        )

        self.timeout = float(
            os.getenv(
                "MODAL_TIMEOUT_SECONDS",
                "120",
            )
        )

        if not self.base_url:
            raise ValueError(
                "MODAL_ENDPOINT_URL is not configured."
            )

    @property
    def headers(self) -> dict[str, str]:

        headers = {
            "Content-Type": "application/json",
        }

        if self.api_key:
            headers["Authorization"] = (
                f"Bearer {self.api_key}"
            )

        return headers

    async def generate(
        self,
        prompt: str,
        max_new_tokens: int,
        temperature: float,
        top_p: float,
    ) -> dict[str, Any]:
        """
        Génération générique.
        """

        payload = {
            "prompt": prompt,
            "max_new_tokens": max_new_tokens,
            "temperature": temperature,
            "top_p": top_p,
        }

        async with httpx.AsyncClient(
            timeout=self.timeout
        ) as client:

            response = await client.post(
                f"{self.base_url}/generate",
                json=payload,
                headers=self.headers,
            )

            response.raise_for_status()

            return response.json()

    async def triage(
        self,
        symptoms: str,
        medical_history: str | None,
        age: int | None,
        priority_context: str | None,
    ) -> dict[str, Any]:
        """
        Endpoint spécialisé triage.
        """

        payload = {
            "symptoms": symptoms,
            "medical_history": medical_history,
            "age": age,
            "priority_context": priority_context,
        }

        async with httpx.AsyncClient(
            timeout=self.timeout
        ) as client:

            response = await client.post(
                f"{self.base_url}/triage",
                json=payload,
                headers=self.headers,
            )

            response.raise_for_status()

            return response.json()


def get_modal_client() -> ModalInferenceClient:
    """
    Factory FastAPI dependency.
    """

    return ModalInferenceClient()
