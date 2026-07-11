# medical-triage-agent-ai-poc/backend/app/api/middleware/auth_middleware.py

from fastapi import Request
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.responses import JSONResponse
from starlette.routing import Match

from backend.app.core.security import verify_access_token


EXCLUDED_PATHS = {"/", "/docs", "/redoc", "/openapi.json", "/health"}


def _route_exists(request: Request) -> bool:
    """
    Returns True if the incoming request matches a registered route
    (regardless of HTTP method), so that unknown paths can fall through
    to FastAPI's routing layer and return a proper 404 instead of a 401.
    """
    for route in request.app.router.routes:
        match, _ = route.matches(request.scope)
        if match != Match.NONE:
            return True
    return False


class JWTAuthMiddleware(BaseHTTPMiddleware):

    async def dispatch(self, request: Request, call_next):

        if request.url.path in EXCLUDED_PATHS:
            return await call_next(request)

        # Let FastAPI's router handle unknown routes -> proper 404,
        # instead of masking them behind a 401 from this middleware.
        if not _route_exists(request):
            return await call_next(request)

        auth_header = request.headers.get("Authorization")

        if not auth_header:
            return JSONResponse(
                status_code=401,
                content={"detail": "Authorization header missing"},
            )

        try:
            scheme, token = auth_header.split()
        except ValueError:
            return JSONResponse(
                status_code=401,
                content={"detail": "Malformed Authorization header"},
            )

        if scheme.lower() != "bearer":
            return JSONResponse(
                status_code=401,
                content={"detail": "Invalid authentication scheme"},
            )

        try:
            verify_access_token(token)
        except Exception:
            return JSONResponse(
                status_code=401,
                content={"detail": "Invalid or expired JWT token"},
            )

        return await call_next(request)
