# medical-triage-agent-ai-poc/backend/app/core/security.py

from datetime import datetime, timedelta, timezone
from typing import Optional

from jose import jwt, ExpiredSignatureError, JWTError
from fastapi import HTTPException, status
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
    """Contenu décodé d'un JWT valide."""

    sub: str
    exp: int


# =========================================================
# CREATION / VERIFICATION DU JWT
# =========================================================


def create_access_token(subject: str, expires_delta: Optional[timedelta] = None) -> str:

    delta = expires_delta or timedelta(minutes=settings.ACCESS_TOKEN_EXPIRE_MINUTES)

    expire = datetime.now(timezone.utc) + delta

    payload = {"sub": subject, "exp": expire}

    encoded_jwt = jwt.encode(
        payload, settings.SECRET_KEY, algorithm=settings.JWT_ALGORITHM
    )

    return encoded_jwt


def create_token_response(
    subject: str, expires_delta: Optional[timedelta] = None
) -> Token:
    """
    Helper utilisé par l'endpoint /auth/token : encapsule
    create_access_token() dans le modèle de réponse Token, avec la
    durée de validité effective (en secondes) pour information client.
    """

    delta = expires_delta or timedelta(minutes=settings.ACCESS_TOKEN_EXPIRE_MINUTES)

    access_token = create_access_token(subject, expires_delta=delta)

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
