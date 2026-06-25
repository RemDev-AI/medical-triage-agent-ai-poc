# medical-triage-agent-ai-poc/backend/app/api/dependencies/inference.py

from __future__ import annotations

import os
from typing import Any

import httpx


class InferenceClient:
    """
    Client HTTP vers le backend d'inférence.

    Responsabilités :
    - envoyer les requêtes d'inférence
    - gérer les timeouts
    - centraliser l'authentification
    - uniformiser les appels HTTP

    Architecture cible :

    Google Colab
        ↓
    Fine-Tuning LoRA / PEFT
        ↓
    Hugging Face Hub
        ↓
    RemDev-AI/medical-triage-agent-ai-poc-models
        ↓
    RemDev-AI/medical-triage-agent-ai-poc-api
        ↓
    InferenceClient
    """

    def __init__(self) -> None:

        self.base_url = os.getenv(
            "INFERENCE_API_URL",
            "",
        ).rstrip("/")

        self.api_token = os.getenv(
            "HF_API_TOKEN",
            "",
        )

        self.timeout = float(
            os.getenv(
                "INFERENCE_TIMEOUT_SECONDS",
                "120",
            )
        )

        if not self.base_url:
            raise ValueError(
                "INFERENCE_API_URL is not configured."
            )

    @property
    def headers(self) -> dict[str, str]:

        headers = {
            "Content-Type": "application/json",
        }

        if self.api_token:
            headers["Authorization"] = (
                f"Bearer {self.api_token}"
            )

        return headers

    async def _post(
        self,
        endpoint: str,
        payload: dict[str, Any],
    ) -> dict[str, Any]:
        """
        Exécute une requête POST vers le backend
        d'inférence.
        """

        async with httpx.AsyncClient(
            timeout=self.timeout,
        ) as client:

            response = await client.post(
                f"{self.base_url}/{endpoint.lstrip('/')}",
                json=payload,
                headers=self.headers,
            )

            response.raise_for_status()

            return response.json()

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

        return await self._post(
            endpoint="generate",
            payload=payload,
        )

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

        return await self._post(
            endpoint="triage",
            payload=payload,
        )


def get_inference_client() -> InferenceClient:
    """
    Factory FastAPI dependency.
    """

    return InferenceClient()
