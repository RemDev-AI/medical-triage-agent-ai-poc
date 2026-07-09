# medical-triage-agent-ai-poc/backend/app/tests/unit/test_api.py

from fastapi.testclient import TestClient

from backend.app.main import app

client = TestClient(app)


def test_health_endpoint():
    """
    Health endpoint should be reachable.
    """

    response = client.get("/health")

    assert response.status_code == 200

    payload = response.json()

    assert isinstance(payload, dict)


def test_openapi_available():
    """
    OpenAPI specification must be exposed.
    """

    response = client.get("/openapi.json")

    assert response.status_code == 200


def test_docs_available():
    """
    Swagger UI must be accessible.
    """

    response = client.get("/docs")

    assert response.status_code == 200


def test_unknown_route_returns_404():
    """
    Unknown endpoint should return 404.
    """

    response = client.get("/unknown-endpoint")

    assert response.status_code == 404


def test_api_returns_json():
    """
    Health endpoint must return JSON.
    """

    response = client.get("/health")

    assert "application/json" in response.headers.get("content-type", "")
