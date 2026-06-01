# medical-triage-agent-ai-poc/backend/app/tests/security/test_endpoints.py

from fastapi.testclient import TestClient

from app.main import app

client = TestClient(app)


def test_unknown_endpoint_returns_404():
    """
    Unknown routes must return 404.
    """

    response = client.get("/not-existing-endpoint")

    assert response.status_code == 404


def test_method_not_allowed():
    """
    Invalid HTTP methods should be rejected.
    """

    response = client.delete("/health")

    assert response.status_code in [404, 405]


def test_invalid_content_type():
    """
    Invalid content-type should fail validation.
    """

    response = client.post(
        "/triage",
        data="invalid",
        headers={
            "Content-Type": "text/plain",
        },
    )

    assert response.status_code in [400, 415, 422]


def test_empty_request_body():
    """
    Empty body should fail schema validation.
    """

    response = client.post(
        "/triage",
        json={},
    )

    assert response.status_code in [400, 422]


def test_openapi_schema_accessible():
    """
    OpenAPI schema should remain accessible.
    """

    response = client.get("/openapi.json")

    assert response.status_code == 200


def test_docs_accessible():
    """
    Swagger UI should remain accessible.
    """

    response = client.get("/docs")

    assert response.status_code == 200
