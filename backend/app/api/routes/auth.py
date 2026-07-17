# medical-triage-agent-ai-poc/backend/app/api/routes/auth.py

"""
Endpoint d'émission de jetons JWT.

Ce fichier comble le trou identifié dans l'architecture : sans lui,
create_access_token()/verify_access_token() existent mais aucun client
(y compris Swagger) ne peut obtenir un jeton valide.

NOTE IMPORTANTE (POC) :
Il n'y a pas encore de base d'utilisateurs / mot de passe. Le POC
vise à protéger l'accès aux endpoints (clé SECRET_KEY côté serveur),
pas à authentifier des utilisateurs individuels. L'endpoint accepte
donc un simple identifiant "client_id" et délivre un jeton, à
condition de fournir un secret partagé (API_ACCESS_KEY) distinct de
SECRET_KEY, pour éviter qu'un tiers non autorisé ne génère lui-même
des jetons simplement en connaissant l'URL.

Si un vrai système utilisateur est introduit plus tard (comptes,
mots de passe, rôles), remplacer TokenRequest / la vérification par
un formulaire OAuth2PasswordRequestForm classique.
"""

from fastapi import APIRouter, HTTPException, status
from pydantic import BaseModel, Field

from app.core.config import settings
from app.core.security import Token, create_token_response

router = APIRouter(prefix="/auth", tags=["Auth"])


class TokenRequest(BaseModel):
    client_id: str = Field(
        ..., min_length=1, description="Identifiant du client demandant un jeton"
    )
    access_key: str = Field(
        ..., min_length=1, description="Clé d'accès partagée (API_ACCESS_KEY)"
    )


@router.post("/token", response_model=Token, summary="Obtenir un jeton JWT d'accès")
async def issue_token(payload: TokenRequest) -> Token:
    """
    Emet un JWT signé (HS256) permettant d'appeler les endpoints
    protégés (ex. POST /triage) via 'Authorization: Bearer <jwt>'.

    Nécessite de fournir la clé d'accès configurée côté serveur
    (variable d'environnement API_ACCESS_KEY), afin que la génération
    de jetons ne soit pas ouverte à n'importe qui connaissant
    simplement l'URL du Space.
    """

    if payload.access_key != settings.API_ACCESS_KEY:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid access key",
        )

    return create_token_response(subject=payload.client_id)
