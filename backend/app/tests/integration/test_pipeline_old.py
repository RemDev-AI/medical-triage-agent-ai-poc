# medical-triage-agent-ai-poc/backend/app/tests/integration/test_pipeline.py

import pytest  # noqa : F401
from fastapi.testclient import TestClient

from backend.app.main import app

client = TestClient(app)


def test_full_triage_pipeline():
    """
    Test end-to-end triage pipeline from input symptoms to recommendations.
    """

    payload = {
        "symptoms": "fièvre, toux, maux de tête",
        "age": 40,
        "medical_history": "diabète type 2"
    }

    response = client.post("/triage", json=payload)

    assert response.status_code == 200
    data = response.json()
    assert "priority" in data
    assert "recommendation" in data
    assert data["priority"] in ["low", "medium", "high", "urgent"]


def test_pipeline_handles_missing_fields():
    """
    Missing optional fields should not break the pipeline.
    """

    payload = {"symptoms": "douleur abdominale"}

    response = client.post("/triage", json=payload)

    assert response.status_code == 200
    data = response.json()
    assert "priority" in data
    assert "recommendation" in data
