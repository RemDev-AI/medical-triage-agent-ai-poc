# medical-triage-agent-ai-poc/backend/app/api/middleware/auth_middleware.py

from fastapi import Request
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.responses import JSONResponse
from starlette.routing import Match

from app.core.security import verify_access_token

# Chemins publics : docs, healthcheck, racine, et désormais /auth/token
# (impossible d'obtenir un jeton si /auth/token est lui-même protégé
# par le même jeton).
EXCLUDED_PATHS = {"/", "/docs", "/redoc", "/openapi.json", "/health", "/auth/token"}


def _route_match(request: Request):
    """
    Inspecte les routes enregistrées et retourne :
      - "full"    : un handler existe pour ce chemin ET cette méthode HTTP
      - "partial" : le chemin existe mais pas pour cette méthode
                    (=> 405 attendu, pas 401)
      - "none"    : le chemin n'existe pas du tout (=> 404 attendu)

    Avant, seul Match.NONE était distingué : un chemin existant mais
    appelé avec la mauvaise méthode (Match.PARTIAL) tombait dans le
    même cas que Match.FULL et se voyait donc à tort exiger un JWT
    avant même que FastAPI ne puisse répondre 405.
    """
    best = Match.NONE

    for route in request.app.router.routes:
        match, _ = route.matches(request.scope)

        if match == Match.FULL:
            return "full"

        if match == Match.PARTIAL and best != Match.FULL:
            best = Match.PARTIAL

    return "partial" if best == Match.PARTIAL else "none"


class JWTAuthMiddleware(BaseHTTPMiddleware):

    async def dispatch(self, request: Request, call_next):

        if request.url.path in EXCLUDED_PATHS:
            return await call_next(request)

        route_match = _route_match(request)

        # Chemin inconnu (404) ou méthode non supportée sur un chemin
        # connu (405) : on laisse FastAPI gérer la réponse plutôt que
        # de masquer l'erreur derrière un 401 trompeur.
        if route_match in ("none", "partial"):
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

        from fastapi import HTTPException

        try:
            verify_access_token(token)

        except HTTPException as exc:
            return JSONResponse(
                status_code=exc.status_code,
                content={"detail": exc.detail},
                headers=exc.headers,
            )

        #         try:
        #             verify_access_token(token)
        #         except Exception:
        #             return JSONResponse(
        #                 status_code=401,
        #                 content={"detail": "Invalid or expired JWT token"},
        #             )

        return await call_next(request)
