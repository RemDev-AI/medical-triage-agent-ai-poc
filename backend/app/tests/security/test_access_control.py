# medical-triage-agent-ai-poc/backend/app/tests/security/test_access_control.py

from fastapi.testclient import TestClient

from app.main import app

client = TestClient(app)


def test_missing_jwt_token():
    """
    Protected endpoints should reject requests
    without authentication.
    """

    response = client.get("/audit")

    assert response.status_code in [401, 403]


def test_invalid_jwt_token():
    """
    Invalid JWT should be rejected.
    """

    headers = {
        "Authorization": "Bearer invalid-token"
    }

    response = client.get(
        "/audit",
        headers=headers,
    )

    assert response.status_code in [401, 403]


def test_malformed_authorization_header():
    """
    Malformed authorization header should fail.
    """

    headers = {
        "Authorization": "InvalidHeader"
    }

    response = client.get(
        "/audit",
        headers=headers,
    )

    assert response.status_code in [401, 403]


def test_access_protected_route_without_role():
    """
    Verify access control is enforced.
    """

    headers = {
        "Authorization": "Bearer fake-user-token"
    }

    response = client.get(
        "/admin",
        headers=headers,
    )

    assert response.status_code in [401, 403, 404]


def test_health_endpoint_public():
    """
    Health endpoint must remain public.
    """

    response = client.get("/health")

    assert response.status_code == 200
