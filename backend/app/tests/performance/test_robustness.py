# medical-triage-agent-ai-poc/backend/app/tests/performance/test_robustness.py

"""
Tests de robustesse (étape 3, point 3 du cahier
des charges) :

    "Réaliser des tests de [...] robustesse [...]."

Couvre :
- validation stricte des entrées (schémas Pydantic)
- comportement en cas de panne du backend
  d'inférence (timeout, exception)
- non-régression sur le double comptage
  request_tracker (correctif étape 3)
- absence de crash sur gpu_monitor.increment_request
  (correctif étape 3, bug précédemment bloquant)
"""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from backend.app.main import app
from backend.app.api.dependencies.inference import (
    get_inference_client,
)
from backend.app.core.security import (
    create_access_token,
)
from backend.app.monitoring.gpu_monitor import (
    gpu_monitor,
)
from backend.app.monitoring.request_tracker import (
    request_tracker,
)
from backend.app.monitoring.alerting import (
    alert_manager,
)


class _FailingInferenceClient:
    """
    Simule une panne du backend d'inférence
    (timeout, service indisponible, etc.).
    """

    async def generate(self, **kwargs):
        raise TimeoutError("Simulated inference backend timeout")

    async def triage(self, **kwargs):
        raise ConnectionError("Simulated inference backend " "unavailable")


class _WorkingInferenceClient:

    async def generate(self, **kwargs):
        return {
            "generated_text": "ok",
            "model_name": "test-model",
            "timestamp": "2026-07-08T00:00:00",
        }

    async def triage(self, **kwargs):
        return {
            "priority_level": "FAIBLE",
            "justification": "ok",
            "recommendations": ["ok"],
            "confidence_score": 0.9,
            "generated_at": "2026-07-08T00:00:00",
        }


@pytest.fixture()
def auth_headers():

    token = create_access_token(subject="robustness-test-user")

    return {"Authorization": f"Bearer {token}"}


@pytest.fixture()
def client_with_failure():

    app.dependency_overrides[get_inference_client] = lambda: _FailingInferenceClient()

    with TestClient(app) as test_client:
        yield test_client

    app.dependency_overrides.clear()


@pytest.fixture()
def client_healthy():

    app.dependency_overrides[get_inference_client] = lambda: _WorkingInferenceClient()

    with TestClient(app) as test_client:
        yield test_client

    app.dependency_overrides.clear()


# ----------------------------------------------------
# Validation des entrées
# ----------------------------------------------------


def test_triage_rejects_empty_symptoms(
    client_healthy,
    auth_headers,
):

    response = client_healthy.post(
        "/triage/",
        json={"symptoms": []},
        headers=auth_headers,
    )

    assert response.status_code == 422


def test_triage_rejects_invalid_age(
    client_healthy,
    auth_headers,
):

    response = client_healthy.post(
        "/triage/",
        json={
            "symptoms": ["fièvre"],
            "age": 250,
        },
        headers=auth_headers,
    )

    assert response.status_code == 422


def test_generate_rejects_oversized_prompt(
    client_healthy,
    auth_headers,
):

    response = client_healthy.post(
        "/generate/",
        json={"prompt": "a" * 10_000},
        headers=auth_headers,
    )

    assert response.status_code == 422


def test_requests_without_token_are_rejected(
    client_healthy,
):

    response = client_healthy.post(
        "/triage/",
        json={"symptoms": ["fièvre"]},
    )

    assert response.status_code == 401


# ----------------------------------------------------
# Pannes du backend d'inférence
# ----------------------------------------------------


def test_triage_failure_returns_500_not_crash(
    client_with_failure,
    auth_headers,
):

    response = client_with_failure.post(
        "/triage/",
        json={"symptoms": ["fièvre"]},
        headers=auth_headers,
    )

    assert response.status_code == 500

    assert "detail" in response.json()


def test_generate_failure_returns_500_not_crash(
    client_with_failure,
    auth_headers,
):

    response = client_with_failure.post(
        "/generate/",
        json={"prompt": "test de panne"},
        headers=auth_headers,
    )

    assert response.status_code == 500


def test_inference_failure_raises_alert(
    client_with_failure,
    auth_headers,
):

    alert_manager.clear()

    client_with_failure.post(
        "/generate/",
        json={"prompt": "test alerte"},
        headers=auth_headers,
    )

    alerts = alert_manager.get_alerts()

    codes = [a["code"] for a in alerts]

    assert "INFERENCE_ERROR" in codes


# ----------------------------------------------------
# Non-régression : compteurs de trafic
# ----------------------------------------------------


def test_request_tracker_is_not_double_counted(
    client_healthy,
    auth_headers,
):
    """
    Correctif étape 3 : vérifie qu'une seule
    incrémentation de total_requests a lieu par
    requête HTTP (middleware uniquement), et non
    deux (middleware + route).
    """

    request_tracker.reset()

    client_healthy.post(
        "/triage/",
        json={"symptoms": ["fièvre"]},
        headers=auth_headers,
    )

    stats = request_tracker.get_stats()

    assert stats["total_requests"] == 1


# ----------------------------------------------------
# Non-régression : gpu_monitor.increment_request
# ----------------------------------------------------


def test_gpu_monitor_increment_request_exists():
    """
    Correctif étape 3 : gpu_monitor.increment_request()
    doit exister et être appelable sans lever
    d'exception (bug bloquant précédemment identifié
    dans TriageEngine.run_triage()).
    """

    before = gpu_monitor.get_request_count()

    gpu_monitor.increment_request()

    after = gpu_monitor.get_request_count()

    assert after == before + 1
