# medical-triage-agent-ai-poc/backend/app/tests/security/conftest.py

import pytest

from backend.app.core.security import create_access_token


@pytest.fixture
def auth_headers() -> dict:
    """
    Provides a valid Authorization header for tests that need to pass
    the JWTAuthMiddleware in order to reach the actual endpoint logic
    (payload validation, injection handling, etc.).
    """
    token = create_access_token(subject="test-user")
    return {"Authorization": f"Bearer {token}"}
