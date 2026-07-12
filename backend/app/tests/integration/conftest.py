# medical-triage-agent-ai-poc/backend/app/tests/integration/conftest.py

import pytest

from backend.app.core.security import create_access_token
from backend.app.main import app
from backend.app.api.dependencies.inference import get_inference_client


@pytest.fixture
def auth_headers() -> dict:
    """
    Fournit un header Authorization JWT valide pour passer le
    JWTAuthMiddleware et atteindre la logique métier réelle des
    endpoints /triage et /generate.
    """
    token = create_access_token(subject="test-user")
    return {"Authorization": f"Bearer {token}"}


class FakeInferenceClient:
    """
    Double de test pour InferenceClient.

    Signatures alignées sur backend/app/api/dependencies/inference.py :
    - generate(prompt, max_new_tokens, temperature, top_p) -> dict avec
      "generated_text" (consommé par GenerateResponse.generated_text)
    - triage(symptoms, medical_history, age, priority_context) -> dict
      avec les clés attendues par la logique de la route triage
    """

    async def generate(self, prompt, max_new_tokens, temperature, top_p):
        return {
            "generated_text": "Simulated generated text for testing purposes.",
            "model_name": "Qwen3-Medical-Triage-Fake",
            "timestamp": "2026-07-12T00:00:00",
        }

    async def triage(self, symptoms, medical_history, age, priority_context):
        return {
            "priority_level": "low",
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
    appel réseau réel vers INFERENCE_API_URL pendant les tests
    d'intégration.
    """
    app.dependency_overrides[get_inference_client] = lambda: FakeInferenceClient()
    yield
    app.dependency_overrides.pop(get_inference_client, None)
