# medical-triage-agent-ai-poc/backend/app/api/middleware/auth_middleware.py

from fastapi import Request, HTTPException
from starlette.middleware.base import BaseHTTPMiddleware

from backend.app.core.security import verify_access_token


EXCLUDED_PATHS = {"/", "/docs", "/redoc", "/openapi.json", "/health/"}


class JWTAuthMiddleware(BaseHTTPMiddleware):

    async def dispatch(self, request: Request, call_next):

        if request.url.path in EXCLUDED_PATHS:
            return await call_next(request)

        auth_header = request.headers.get("Authorization")

        if not auth_header:
            raise HTTPException(status_code=401, detail="Authorization header missing")

        try:
            scheme, token = auth_header.split()

            if scheme.lower() != "bearer":
                raise HTTPException(
                    status_code=401, detail="Invalid authentication scheme"
                )

            verify_access_token(token)

        except ValueError:
            raise HTTPException(
                status_code=401, detail="Malformed Authorization header"
            )

        return await call_next(request)
