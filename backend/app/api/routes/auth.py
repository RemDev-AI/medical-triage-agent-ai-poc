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
import secrets

from app.core.config import settings
from app.core.security import Token, create_token_response

router = APIRouter(prefix="/auth", tags=["Auth"])


class TokenRequest(BaseModel):
    client_id: str = Field(
        ..., min_length=1, description="Identifiant du client demandant un jeton"
    )
    access_key: str = Field(
        ...,
        min_length=1,
        description=(
            "Clé d'accès partagée. Utiliser API_ACCESS_KEY pour un "
            "jeton scope='triage' (soumission de triages), ou "
            "AUDIT_ACCESS_KEY pour un jeton scope='audit' (lecture "
            "de GET /audit/ et GET /audit/clinical)."
        ),
    )


def _resolve_scope_for_access_key(access_key: str) -> str:
    """
    Détermine le scope à attribuer au JWT émis, en fonction de QUEL
    secret configuré correspond à la clé présentée.

    Volontairement PAS un champ "scope" fourni par le client dans
    TokenRequest : le scope doit découler d'un secret que seul un
    porteur autorisé connaît, jamais d'une simple déclaration côté
    client (qui pourrait sinon s'auto-attribuer scope="audit" avec
    la même clé standard API_ACCESS_KEY).

    Comparaisons en temps constant (secrets.compare_digest) plutôt
    que "==" : ces secrets protègent, in fine, l'accès à des données
    patient (via scope="audit") — une comparaison naïve est
    vulnérable à une attaque par mesure de temps, même si le risque
    reste théorique pour un POC.
    """

    if secrets.compare_digest(access_key, settings.API_ACCESS_KEY):
        return "triage"

    if settings.AUDIT_ACCESS_KEY and secrets.compare_digest(
        access_key, settings.AUDIT_ACCESS_KEY
    ):
        return "audit"

    return ""


@router.post("/token", response_model=Token, summary="Obtenir un jeton JWT d'accès")
async def issue_token(payload: TokenRequest) -> Token:
    """
    Emet un JWT signé (HS256) permettant d'appeler les endpoints
    protégés via 'Authorization: Bearer <jwt>'.

    Le scope du jeton émis (donc les endpoints accessibles) dépend
    de la clé d'accès fournie :
    - API_ACCESS_KEY  -> scope="triage"  (ex: POST /triage/)
    - AUDIT_ACCESS_KEY -> scope="audit" (ex: GET /audit/,
      GET /audit/clinical — données patient, à réserver au
      personnel autorisé)

    Nécessite de fournir l'une de ces clés configurées côté serveur,
    afin que la génération de jetons ne soit pas ouverte à n'importe
    qui connaissant simplement l'URL du Space.
    """

    scope = _resolve_scope_for_access_key(payload.access_key)

    if not scope:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid access key",
        )

    return create_token_response(subject=payload.client_id, scope=scope)
