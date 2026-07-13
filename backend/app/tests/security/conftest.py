# medical-triage-agent-ai-poc/backend/app/tests/security/conftest.py

import pytest

from app.core.security import create_access_token
from app.main import app
from app.api.dependencies.inference import get_inference_client


@pytest.fixture
def auth_headers() -> dict:
    """
    Provides a valid Authorization header for tests that need to pass
    the JWTAuthMiddleware in order to reach the actual endpoint logic
    (payload validation, injection handling, etc.).
    """
    token = create_access_token(subject="test-user")
    return {"Authorization": f"Bearer {token}"}


class FakeInferenceClient:
    """
    Double de test pour InferenceClient.

    Les tests de sécurité (injections, XSS, prompt injection, gros payloads)
    doivent rester déterministes et isolés : ils ne doivent jamais dépendre
    d'un backend d'inférence externe réel (réseau, quotas, filtrage côté
    modèle, etc.). Ce double simule une réponse de triage "propre" quel
    que soit le contenu reçu, ce qui permet de tester exclusivement la
    logique de l'API (validation Pydantic, filtrage anti-fuite,
    gestion d'erreurs) sans bruit externe.
    """

    async def triage(self, symptoms, medical_history, age, priority_context):
        return {
            "priority_level": "LOW",
            "justification": "Simulated triage result for testing purposes.",
            "recommendations": ["Rest and monitor symptoms."],
            "confidence_score": 0.9,
            "generated_at": "2026-07-12T00:00:00",
        }


@pytest.fixture(autouse=True)
def override_inference_client():
    """
    Remplace la dépendance InferenceClient par un double de test pour
    tous les tests de ce dossier (autouse=True), afin d'éviter tout
    appel réseau réel vers INFERENCE_API_URL pendant les tests de
    sécurité.
    """
    app.dependency_overrides[get_inference_client] = lambda: FakeInferenceClient()
    yield
    app.dependency_overrides.pop(get_inference_client, None)
