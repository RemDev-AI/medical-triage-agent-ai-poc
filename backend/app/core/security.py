# medical-triage-agent-ai-poc/backend/app/core/security.py

from datetime import datetime, timedelta, timezone
from typing import Optional

from jose import jwt, ExpiredSignatureError, JWTError
from fastapi import HTTPException, Request, status
from pydantic import BaseModel

from app.core.config import settings

# =========================================================
# MODELES
# =========================================================


class Token(BaseModel):
    """Réponse renvoyée par POST /auth/token."""

    access_token: str
    token_type: str = "bearer"
    expires_in: int  # secondes


class TokenPayload(BaseModel):
    """
    Contenu décodé d'un JWT valide.

    scope : ajouté (2026-07-21) pour distinguer les usages du JWT.
    Jusqu'ici, un JWT valide donnait accès à TOUTES les routes
    protégées, y compris /audit/clinical qui expose désormais des
    données patient (symptômes, antécédents, sortie brute du
    modèle). Un client applicatif qui n'a besoin que de soumettre
    des triages n'a aucune raison de pouvoir aussi consulter ce
    journal.

    Défaut = "triage" (moindre privilège) : tout jeton émis AVANT ce
    correctif (donc sans claim "scope" dans son payload) sera donc
    traité comme non-habilité pour l'audit, plutôt que l'inverse. La
    délivrance de jetons avec scope="audit" reste à câbler dans la
    route POST /auth/token (non fournie à ce stade) — voir note en
    fin de fichier.
    """

    sub: str
    exp: int
    scope: str = "triage"


# =========================================================
# CREATION / VERIFICATION DU JWT
# =========================================================


def create_access_token(
    subject: str,
    expires_delta: Optional[timedelta] = None,
    scope: str = "triage",
) -> str:

    delta = expires_delta or timedelta(minutes=settings.ACCESS_TOKEN_EXPIRE_MINUTES)

    expire = datetime.now(timezone.utc) + delta

    payload = {"sub": subject, "exp": expire, "scope": scope}

    encoded_jwt = jwt.encode(
        payload, settings.SECRET_KEY, algorithm=settings.JWT_ALGORITHM
    )

    return encoded_jwt


def create_token_response(
    subject: str,
    expires_delta: Optional[timedelta] = None,
    scope: str = "triage",
) -> Token:
    """
    Helper utilisé par l'endpoint /auth/token : encapsule
    create_access_token() dans le modèle de réponse Token, avec la
    durée de validité effective (en secondes) pour information client.
    """

    delta = expires_delta or timedelta(minutes=settings.ACCESS_TOKEN_EXPIRE_MINUTES)

    access_token = create_access_token(subject, expires_delta=delta, scope=scope)

    return Token(access_token=access_token, expires_in=int(delta.total_seconds()))


def verify_access_token(token: str) -> TokenPayload:

    try:
        payload = jwt.decode(
            token, settings.SECRET_KEY, algorithms=[settings.JWT_ALGORITHM]
        )

        return TokenPayload(**payload)

    except ExpiredSignatureError:
        # Distingué explicitement d'un jeton invalide : un jeton expiré
        # est structurellement correct, il a juste besoin d'être
        # renouvelé (POST /auth/token), contrairement à un jeton
        # falsifié ou mal signé.
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Token has expired",
            headers={"WWW-Authenticate": "Bearer"},
        )

    except JWTError:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid JWT token",
            headers={"WWW-Authenticate": "Bearer"},
        )


def require_scope(required_scope: str):
    """
    Factory de dépendance FastAPI pour restreindre une route à un
    scope de JWT donné (ex: "audit"), en plus de l'authentification
    déjà assurée par JWTAuthMiddleware pour toute route non exclue.

    S'appuie sur request.state.token_payload, renseigné par
    JWTAuthMiddleware après une vérification réussie du JWT (évite
    de re-décoder le token une seconde fois ici).

    Usage :
        @router.get("/clinical", dependencies=[Depends(require_scope("audit"))])
    """

    def _dependency(request: Request) -> TokenPayload:

        payload: Optional[TokenPayload] = getattr(request.state, "token_payload", None)

        if payload is None:
            # Ne devrait pas arriver si JWTAuthMiddleware est bien
            # monté et que cette route n'est pas dans EXCLUDED_PATHS
            # — mais on ne fait jamais confiance implicitement à
            # l'ordre de montage des middlewares pour une décision
            # d'autorisation sur des données patient.
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Missing authenticated token context.",
                headers={"WWW-Authenticate": "Bearer"},
            )

        if payload.scope != required_scope:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail=(
                    f"This endpoint requires scope '{required_scope}', "
                    f"token has scope '{payload.scope}'."
                ),
            )

        return payload

    return _dependency
