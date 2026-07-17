# medical-triage-agent-ai-poc/backend/app/tests/security/conftest.py

import pytest

from app.core.security import create_access_token
from app.main import app
from app.api.dependencies.inference import get_triage_engine


@pytest.fixture
def auth_headers() -> dict:
    """
    Provides a valid Authorization header for tests that need to pass
    the JWTAuthMiddleware in order to reach the actual endpoint logic
    (payload validation, injection handling, etc.).
    """
    token = create_access_token(subject="test-user")
    return {"Authorization": f"Bearer {token}"}


class FakeTriageEngine:
    """
    Double de test pour TriageEngine.

    Les tests de sécurité (injections, XSS, prompt injection, gros payloads)
    doivent rester déterministes et isolés : ils ne doivent jamais dépendre
    d'un moteur d'inférence réel (chargement de modèle, GPU, comportement
    non déterministe du LLM, etc.). Ce double simule une réponse de triage
    "propre" quel que soit le contenu reçu, ce qui permet de tester
    exclusivement la logique de l'API (validation Pydantic, filtrage
    anti-fuite, gestion d'erreurs) sans bruit externe.

    Format aligné sur TriageEngine.run_triage() réel :
    {"triage": {...}, "metadata": {...}, "raw_response": ...}.
    """

    async def run_triage(self, **kwargs):
        return {
            "triage": {
                "priority": "FAIBLE",
                "justification": "Simulated triage result for testing purposes.",
                "recommendations": "Rest and monitor symptoms.",
                "confidence_score": 0.9,
            },
            "metadata": {
                "latency_seconds": 0.01,
                "model_name": "test-model",
            },
            "raw_response": "PRIORITÉ:\nFAIBLE\n",
        }


@pytest.fixture(autouse=True)
def override_inference_client():
    """
    Remplace la dépendance TriageEngine par un double de test pour
    tous les tests de ce dossier (autouse=True), afin d'éviter tout
    chargement réel de modèle pendant les tests de sécurité.
    """
    app.dependency_overrides[get_triage_engine] = lambda: FakeTriageEngine()
    yield
    app.dependency_overrides.pop(get_triage_engine, None)
