# medical-triage-agent-ai-poc/backend/app/tests/integration/test_pipeline.py

"""
End-to-end integration tests.

Architecture:

HF Space UI
      ↓
HF Space API
      ↓
Modal Endpoint
      ↓
Qwen3-LoRA

No local model loading.
"""

from fastapi.testclient import TestClient

from backend.app.main import app

client = TestClient(app)


def test_full_triage_pipeline():
    """
    Complete triage workflow.
    """

    payload = {
        "symptoms": "fièvre, toux, maux de tête",
        "age": 40,
        "medical_history": "diabète type 2",
    }

    response = client.post(
        "/triage",
        json=payload,
    )

    assert response.status_code == 200

    data = response.json()

    assert "priority" in data
    assert "recommendation" in data

    assert data["priority"] in [
        "low",
        "medium",
        "high",
        "urgent",
    ]

    assert isinstance(
        data["recommendation"],
        str,
    )


def test_pipeline_handles_missing_fields():
    """
    Optional fields must not break pipeline execution.
    """

    payload = {
        "symptoms": "douleur abdominale",
    }

    response = client.post(
        "/triage",
        json=payload,
    )

    assert response.status_code == 200

    data = response.json()

    assert "priority" in data
    assert "recommendation" in data


def test_pipeline_with_complex_patient_profile():
    """
    Pipeline should support richer patient context.
    """

    payload = {
        "symptoms": (
            "douleur thoracique légère, fatigue,"
            " essoufflement"
        ),
        "age": 68,
        "medical_history": (
            "hypertension, diabète type 2"
        ),
        "medications": [
            "metformine",
            "amlodipine",
        ],
    }

    response = client.post(
        "/triage",
        json=payload,
    )

    assert response.status_code == 200

    data = response.json()

    assert "priority" in data
    assert "recommendation" in data


def test_pipeline_response_contract():
    """
    API response contract validation.
    """

    payload = {
        "symptoms": "nausées persistantes",
        "age": 31,
    }

    response = client.post(
        "/triage",
        json=payload,
    )

    assert response.status_code == 200

    body = response.json()

    required_fields = [
        "priority",
        "recommendation",
    ]

    for field in required_fields:
        assert field in body


def test_pipeline_priority_domain():
    """
    Validate triage classification domain.
    """

    payload = {
        "symptoms": "perte de connaissance",
        "age": 55,
    }

    response = client.post(
        "/triage",
        json=payload,
    )

    assert response.status_code == 200

    data = response.json()

    allowed_priorities = {
        "low",
        "medium",
        "high",
        "urgent",
    }

    assert data["priority"] in allowed_priorities


def test_pipeline_supports_modal_provider():
    """
    Validate provider metadata when available.
    """

    payload = {
        "symptoms": "forte fièvre",
        "age": 48,
    }

    response = client.post(
        "/triage",
        json=payload,
    )

    assert response.status_code == 200

    data = response.json()

    if "provider" in data:
        assert data["provider"] in [
            "modal",
            "fallback",
        ]


def test_pipeline_supports_observability_metadata():
    """
    Validate latency and monitoring metadata.
    """

    payload = {
        "symptoms": "vertiges",
        "age": 42,
    }

    response = client.post(
        "/triage",
        json=payload,
    )

    assert response.status_code == 200

    data = response.json()

    if "latency_ms" in data:
        assert isinstance(
            data["latency_ms"],
            (int, float),
        )

    if "request_id" in data:
        assert isinstance(
            data["request_id"],
            str,
        )


def test_pipeline_handles_large_payload():
    """
    Pipeline should remain stable with large requests.
    """

    payload = {
        "symptoms": "fièvre " * 300,
        "age": 37,
        "medical_history": "asthme",
    }

    response = client.post(
        "/triage",
        json=payload,
    )

    assert response.status_code == 200

    data = response.json()

    assert "priority" in data
    assert "recommendation" in data


def test_pipeline_recommendation_not_empty():
    """
    Recommendation must contain usable content.
    """

    payload = {
        "symptoms": "toux persistante",
        "age": 50,
    }

    response = client.post(
        "/triage",
        json=payload,
    )

    assert response.status_code == 200

    data = response.json()

    assert data["recommendation"].strip() != ""
