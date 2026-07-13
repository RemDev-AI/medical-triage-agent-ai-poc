# medical-triage-agent-ai-poc/backend/app/tests/performance/test_audit_trail.py

"""
Tests d'audit de traçabilité (étape 3, points 3 et
13 du cahier des charges) :

    "Réaliser [...] des audits de traçabilité des
    interactions."
    "Documenter clairement les limites d'usage pour
    les utilisateurs."

Valide le correctif étape 3 : /audit/ est désormais
connecté à un stockage réel (audit_store.py) au lieu
de données mockées en dur.
"""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from app.main import app
from app.api.dependencies.inference import (
    get_inference_client,
)
from app.core.security import (
    create_access_token,
)
from app.monitoring import audit_store


class _FakeInferenceClient:

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
def client():

    app.dependency_overrides[get_inference_client] = lambda: _FakeInferenceClient()

    audit_store.clear()

    with TestClient(app) as test_client:
        yield test_client

    app.dependency_overrides.clear()


@pytest.fixture()
def auth_headers():

    token = create_access_token(subject="audit-test-user")

    return {"Authorization": f"Bearer {token}"}


def test_interaction_is_persisted_to_audit_store(
    client,
    auth_headers,
):

    response = client.post(
        "/triage/",
        json={"symptoms": ["fièvre"]},
        headers=auth_headers,
    )

    assert response.status_code == 200

    entries = audit_store.read_entries()

    assert len(entries) >= 1

    triage_entries = [e for e in entries if e["path"] == "/triage/"]

    assert len(triage_entries) == 1

    entry = triage_entries[0]

    assert entry["method"] == "POST"
    assert entry["status_code"] == 200
    assert "request_id" in entry
    assert "latency_ms" in entry


def test_audit_endpoint_returns_real_data(
    client,
    auth_headers,
):

    client.post(
        "/triage/",
        json={"symptoms": ["fièvre"]},
        headers=auth_headers,
    )

    response = client.get(
        "/audit/",
        headers=auth_headers,
    )

    assert response.status_code == 200

    body = response.json()

    assert body["total_logs"] >= 1

    assert any(log["endpoint"] == "/triage/" for log in body["logs"])


def test_audit_endpoint_respects_limit(
    client,
    auth_headers,
):

    for _ in range(5):

        client.get(
            "/health/",
        )

    response = client.get(
        "/audit/?limit=2",
        headers=auth_headers,
    )

    assert response.status_code == 200

    body = response.json()

    assert len(body["logs"]) <= 2


def test_audit_entries_are_most_recent_first(
    client,
    auth_headers,
):

    client.post(
        "/triage/",
        json={"symptoms": ["premier"]},
        headers=auth_headers,
    )

    client.post(
        "/triage/",
        json={"symptoms": ["second"]},
        headers=auth_headers,
    )

    entries = audit_store.read_entries()

    triage_entries = [e for e in entries if e["path"] == "/triage/"]

    assert len(triage_entries) >= 2

    timestamps = [e["timestamp"] for e in triage_entries]

    assert timestamps == sorted(timestamps, reverse=True)
