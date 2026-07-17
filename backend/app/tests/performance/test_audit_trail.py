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
    get_triage_engine,
    get_generation_context,
)
from app.core.security import (
    create_access_token,
)
from app.monitoring import audit_store


class _FakeTriageEngine:

    async def run_triage(self, **kwargs):
        return {
            "triage": {
                "priority": "FAIBLE",
                "justification": "ok",
                "recommendations": "ok",
                "confidence_score": 0.9,
            },
            "metadata": {
                "latency_seconds": 0.01,
                "model_name": "test-model",
            },
            "raw_response": "ok",
        }


async def _fake_generate_response(**kwargs):
    return "ok"


@pytest.fixture()
def client(monkeypatch):

    app.dependency_overrides[get_triage_engine] = lambda: _FakeTriageEngine()
    app.dependency_overrides[get_generation_context] = lambda: (None, None)

    monkeypatch.setattr(
        "app.api.routes.inference.generate_response",
        _fake_generate_response,
    )

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
        json={"symptoms": "fièvre"},
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
        json={"symptoms": "fièvre"},
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
        json={"symptoms": "premier"},
        headers=auth_headers,
    )

    client.post(
        "/triage/",
        json={"symptoms": "second"},
        headers=auth_headers,
    )

    entries = audit_store.read_entries()

    triage_entries = [e for e in entries if e["path"] == "/triage/"]

    assert len(triage_entries) >= 2

    timestamps = [e["timestamp"] for e in triage_entries]

    assert timestamps == sorted(timestamps, reverse=True)
