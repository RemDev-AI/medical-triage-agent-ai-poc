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


def test_full_triage_pipeline(auth_headers):
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
        headers=auth_headers,
    )

    assert response.status_code == 200

    data = response.json()

    assert "priority_level" in data
    assert "recommendations" in data

    assert data["priority_level"] in [
        "low",
        "medium",
        "high",
        "urgent",
    ]

    assert isinstance(
        data["recommendations"],
        list,
    )


def test_pipeline_handles_missing_fields(auth_headers):
    """
    Optional fields must not break pipeline execution.
    """

    payload = {
        "symptoms": "douleur abdominale",
    }

    response = client.post(
        "/triage",
        json=payload,
        headers=auth_headers,
    )

    assert response.status_code == 200

    data = response.json()

    assert "priority_level" in data
    assert "recommendations" in data


def test_pipeline_with_complex_patient_profile(auth_headers):
    """
    Pipeline should support richer patient context.
    """

    payload = {
        "symptoms": ("douleur thoracique légère, fatigue," " essoufflement"),
        "age": 68,
        "medical_history": ("hypertension, diabète type 2"),
        "medications": [
            "metformine",
            "amlodipine",
        ],
    }

    response = client.post(
        "/triage",
        json=payload,
        headers=auth_headers,
    )

    assert response.status_code == 200

    data = response.json()

    assert "priority_level" in data
    assert "recommendations" in data


def test_pipeline_response_contract(auth_headers):
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
        headers=auth_headers,
    )

    assert response.status_code == 200

    body = response.json()

    required_fields = [
        "priority_level",
        "recommendations",
    ]

    for field in required_fields:
        assert field in body


def test_pipeline_priority_domain(auth_headers):
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
        headers=auth_headers,
    )

    assert response.status_code == 200

    data = response.json()

    allowed_priorities = {
        "low",
        "medium",
        "high",
        "urgent",
    }

    assert data["priority_level"] in allowed_priorities


def test_pipeline_supports_modal_provider(auth_headers):
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
        headers=auth_headers,
    )

    assert response.status_code == 200

    data = response.json()

    if "provider" in data:
        assert data["provider"] in [
            "modal",
            "fallback",
        ]


def test_pipeline_supports_observability_metadata(auth_headers):
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
        headers=auth_headers,
    )

    assert response.status_code == 200

    data = response.json()

    assert isinstance(
        data["latency_seconds"],
        (int, float),
    )

    if "request_id" in data:
        assert isinstance(
            data["request_id"],
            str,
        )


def test_pipeline_handles_large_payload(auth_headers):
    """
    Pipeline should remain stable with large requests.
    """

    payload = {
        "symptoms": "fièvre " * 250,
        "age": 37,
        "medical_history": "asthme",
    }

    response = client.post(
        "/triage",
        json=payload,
        headers=auth_headers,
    )

    assert response.status_code == 200

    data = response.json()

    assert "priority_level" in data
    assert "recommendations" in data


def test_pipeline_recommendation_not_empty(auth_headers):
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
        headers=auth_headers,
    )

    assert response.status_code == 200

    data = response.json()

    assert len(data["recommendations"]) > 0
    assert data["recommendations"][0].strip() != ""
