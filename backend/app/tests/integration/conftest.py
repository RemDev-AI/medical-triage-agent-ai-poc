# medical-triage-agent-ai-poc/backend/app/tests/integration/conftest.py

import pytest

from app.core.security import create_access_token
from app.main import app
from app.api.dependencies.inference import (
    get_triage_engine,
    get_generation_context,
)


@pytest.fixture
def auth_headers() -> dict:
    """
    Fournit un header Authorization JWT valide pour passer le
    JWTAuthMiddleware et atteindre la logique métier réelle des
    endpoints /triage et /generate.
    """
    token = create_access_token(subject="test-user")
    return {"Authorization": f"Bearer {token}"}


class FakeTriageEngine:
    """
    Double de test pour TriageEngine.

    Signature alignée sur backend/app/llm/inference/triage_engine.py :
    run_triage(patient_age, symptoms, medical_history, vital_signs)
    -> dict avec la clé "triage" (consommée par routes/triage.py),
    au même format que TriageEngine.run_triage() réel :
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
                "model_name": "Qwen3-Medical-Triage-Fake",
            },
            "raw_response": "PRIORITÉ:\nFAIBLE\n",
        }


async def _fake_generate_response(**kwargs):
    """
    Double de test pour generate_response(). Contrairement à
    TriageEngine, generate_response n'est pas injecté via Depends
    dans routes/inference.py : on le patche donc directement au
    niveau du module qui l'importe.
    """
    return "Simulated generated text for testing purposes."


@pytest.fixture(autouse=True)
def override_inference_dependencies(monkeypatch):
    """
    Remplace TriageEngine et generate_response par des doubles de
    test pour tous les tests de ce dossier (autouse=True), afin
    d'éviter tout chargement réel de modèle pendant les tests
    d'intégration.
    """
    app.dependency_overrides[get_triage_engine] = lambda: FakeTriageEngine()
    app.dependency_overrides[get_generation_context] = lambda: (None, None)

    monkeypatch.setattr(
        "app.api.routes.inference.generate_response",
        _fake_generate_response,
    )

    yield

    app.dependency_overrides.pop(get_triage_engine, None)
    app.dependency_overrides.pop(get_generation_context, None)
