# medical-triage-agent-ai-poc/backend/app/core/security.py

from datetime import datetime, timedelta
from typing import Optional

from jose import jwt, JWTError
from fastapi import HTTPException, status

from backend.app.core.config import settings


def create_access_token(subject: str, expires_delta: Optional[timedelta] = None) -> str:

    expire = datetime.utcnow() + (
        expires_delta or timedelta(minutes=settings.ACCESS_TOKEN_EXPIRE_MINUTES)
    )

    payload = {"sub": subject, "exp": expire}

    encoded_jwt = jwt.encode(
        payload, settings.SECRET_KEY, algorithm=settings.JWT_ALGORITHM
    )

    return encoded_jwt


def verify_access_token(token: str) -> dict:

    try:
        payload = jwt.decode(
            token, settings.SECRET_KEY, algorithms=[settings.JWT_ALGORITHM]
        )

        return payload

    except JWTError:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid or expired JWT token",
        )
