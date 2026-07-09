# medical-triage-agent-ai-poc/backend/app/tests/performance/test_latency.py

"""
Tests de latence (étape 3, point 9 du cahier des
charges) :

    "Mesurer la latence et le temps de réponse en
    conditions réalistes."

L'InferenceClient (appel HTTP externe vers le
backend d'inférence) est mocké afin de tester le
comportement de l'API elle-même (validation,
monitoring, audit) indépendamment de la
disponibilité d'un GPU ou d'un Space Hugging Face
réel, conformément à l'environnement CI
(ubuntu-latest, sans GPU).
"""

from __future__ import annotations

import asyncio
import statistics  # noqa : F401
import time

import pytest
from fastapi.testclient import TestClient

from backend.app.main import app
from backend.app.api.dependencies.inference import (
    get_inference_client,
)
from backend.app.core.security import (
    create_access_token,
)
from backend.app.monitoring.latency_monitor import (
    latency_monitor,
)


class _FakeInferenceClient:
    """
    Double de test pour InferenceClient, simulant
    une latence d'inférence réaliste (ordre de
    grandeur d'un appel vLLM sur un petit modèle).
    """

    async def generate(self, **kwargs):
        await asyncio.sleep(0.02)
        return {
            "generated_text": "réponse simulée",
            "model_name": "qwen3-1.7b-dpo-test",
            "timestamp": "2026-07-08T00:00:00",
        }

    async def triage(self, **kwargs):
        await asyncio.sleep(0.02)
        return {
            "priority_level": "MODÉRÉ",
            "justification": "justification simulée",
            "recommendations": ["repos", "hydratation"],
            "confidence_score": 0.9,
            "generated_at": "2026-07-08T00:00:00",
        }


@pytest.fixture()
def client():

    app.dependency_overrides[get_inference_client] = lambda: _FakeInferenceClient()

    latency_monitor.reset()

    with TestClient(app) as test_client:
        yield test_client

    app.dependency_overrides.clear()


@pytest.fixture()
def auth_headers():

    token = create_access_token(subject="latency-test-user")

    return {"Authorization": f"Bearer {token}"}


# Seuils alignés sur AlertManager.LATENCY_WARNING_MS
# (backend/app/monitoring/alerting.py), afin que les
# tests et les alertes de production partagent la
# même référence.
LATENCY_WARNING_MS = 1000
ACCEPTABLE_P95_MS = 500


def test_triage_endpoint_latency_is_measured(
    client,
    auth_headers,
):
    """
    Vérifie que /triage/ répond avec un
    latency_seconds cohérent et sous le seuil
    d'alerte WARNING.
    """

    payload = {
        "symptoms": ["fièvre", "toux"],
        "age": 34,
    }

    start = time.perf_counter()

    response = client.post(
        "/triage/",
        json=payload,
        headers=auth_headers,
    )

    elapsed_ms = (time.perf_counter() - start) * 1000

    assert response.status_code == 200

    body = response.json()

    assert "latency_seconds" in body

    assert body["latency_seconds"] >= 0

    assert elapsed_ms < LATENCY_WARNING_MS, (
        f"Latence observée trop élevée : " f"{elapsed_ms:.2f} ms"
    )


def test_generate_endpoint_latency_under_load(
    client,
    auth_headers,
):
    """
    Simule une charge réaliste (N requêtes
    séquentielles) et vérifie la distribution de
    latence (p95) via latency_monitor, alimenté par
    AuditLoggingMiddleware.
    """

    payload = {
        "prompt": "Décris un cas clinique standard.",
    }

    n_requests = 20

    for _ in range(n_requests):

        response = client.post(
            "/generate/",
            json=payload,
            headers=auth_headers,
        )

        assert response.status_code == 200

    stats = latency_monitor.stats()

    assert stats["count"] >= n_requests

    assert stats["p95_ms"] < ACCEPTABLE_P95_MS, (
        f"p95 de latence trop élevé : " f"{stats['p95_ms']} ms"
    )


def test_latency_stats_endpoint_reflects_traffic(
    client,
    auth_headers,
):
    """
    Vérifie que /monitoring/latency reflète le
    trafic généré, pour usage dans un dashboard ou
    un rapport de tests de charge.
    """

    for _ in range(5):

        client.post(
            "/triage/",
            json={
                "symptoms": ["douleur thoracique"],
            },
            headers=auth_headers,
        )

    response = client.get(
        "/monitoring/latency",
        headers=auth_headers,
    )

    assert response.status_code == 200

    metrics = response.json()["metrics"]

    assert metrics["count"] >= 5
    assert metrics["avg_ms"] >= 0
