# medical-triage-agent-ai-poc/frontend/streamlit_app/services/auth_client.py

"""
Gestion du cycle de vie du JWT côté frontend Streamlit.

Rôle :
  - obtenir un JWT auprès de POST /auth/token (backend) ;
  - le garder en cache dans st.session_state pour éviter une
    requête réseau à chaque interaction utilisateur ;
  - le renouveler automatiquement un peu avant son expiration
    (marge de sécurité), de façon totalement transparente pour
    le reste de l'application ;
  - exposer un header prêt à l'emploi pour tout appel API
    protégé (ex. POST /triage, GET /monitoring/overview).

Ce module ne connaît jamais SECRET_KEY (clé de signature, réservée
au backend) : il ne fait que consommer le jeton émis par l'API et
lire sa date d'expiration (claim "exp"), sans vérifier la signature
- ce n'est pas son rôle, seul le backend vérifie/valide le JWT.

NOTE (alignement architecture) :
La configuration est lue depuis streamlit_app/config/settings.py
(qui la source elle-même depuis hf_space_config.py), au même titre
que API_BASE_URL / REQUEST_TIMEOUT déjà utilisés par api_client.py,
triage_api.py et metrics_api.py — plutôt que de relire os.environ
directement ici, ce qui aurait introduit une deuxième source de
vérité pour les mêmes variables.
"""

from __future__ import annotations

import base64
import json
import time
from typing import Optional

import requests
import streamlit as st

from streamlit_app.config.settings import (
    API_ACCESS_KEY,
    API_BASE_URL,
    REQUEST_TIMEOUT,
    STREAMLIT_CLIENT_ID,
)

# Marge de sécurité (secondes) : le jeton est renouvelé dès qu'il lui
# reste moins de ce délai avant expiration, pour éviter qu'une requête
# en cours n'échoue pile au moment du renouvellement.
TOKEN_REFRESH_MARGIN_SECONDS = 30

# Clés utilisées dans st.session_state
_SESSION_KEY_TOKEN = "_auth_access_token"
_SESSION_KEY_EXPIRES_AT = "_auth_token_expires_at"


class AuthError(Exception):
    """Levée si l'obtention d'un JWT auprès du backend échoue."""


# =========================================================
# DECODAGE DU CLAIM "exp" (sans vérification de signature)
# =========================================================


def _decode_exp_claim(jwt_token: str) -> Optional[int]:
    """
    Extrait le timestamp d'expiration (claim "exp") d'un JWT sans
    vérifier sa signature : le frontend ne possède pas SECRET_KEY,
    il n'a pas à valider le jeton, seulement à savoir quand le
    renouveler proactivement.
    """
    try:
        payload_segment = jwt_token.split(".")[1]
        # Le padding base64url doit être un multiple de 4
        padding = "=" * (-len(payload_segment) % 4)
        decoded = base64.urlsafe_b64decode(payload_segment + padding)
        claims = json.loads(decoded)
        return int(claims.get("exp")) if "exp" in claims else None
    except Exception:
        return None


# =========================================================
# OBTENTION D'UN NOUVEAU JETON
# =========================================================


def _request_new_token() -> str:
    if not API_ACCESS_KEY:
        raise AuthError(
            "API_ACCESS_KEY n'est pas configurée côté frontend "
            "(Secret du Space UI manquant — cf. hf_space_config.py)."
        )

    try:
        response = requests.post(
            f"{API_BASE_URL}/auth/token",
            json={"client_id": STREAMLIT_CLIENT_ID, "access_key": API_ACCESS_KEY},
            timeout=REQUEST_TIMEOUT,
        )
    except requests.RequestException as exc:
        raise AuthError(
            f"Impossible de joindre {API_BASE_URL}/auth/token : {exc}"
        ) from exc

    if response.status_code != 200:
        raise AuthError(
            f"Echec d'obtention du jeton (HTTP {response.status_code}) : {response.text}"
        )

    data = response.json()
    access_token = data.get("access_token")

    if not access_token:
        raise AuthError("Réponse de /auth/token sans champ 'access_token'.")

    return access_token


# =========================================================
# API PUBLIQUE DU MODULE
# =========================================================


def get_valid_token() -> str:
    """
    Retourne un JWT valide, en le réutilisant depuis
    st.session_state s'il est encore valable, ou en le renouvelant
    automatiquement sinon (premier appel, expiration proche/dépassée).

    Lève AuthError si le backend ne peut pas délivrer de jeton.
    """

    cached_token = st.session_state.get(_SESSION_KEY_TOKEN)
    cached_expires_at = st.session_state.get(_SESSION_KEY_EXPIRES_AT)

    now = time.time()

    if (
        cached_token
        and cached_expires_at
        and cached_expires_at - now > TOKEN_REFRESH_MARGIN_SECONDS
    ):
        return cached_token

    # Jeton absent, expiré, ou sur le point d'expirer : on en obtient un nouveau.
    new_token = _request_new_token()
    exp_ts = _decode_exp_claim(new_token)

    st.session_state[_SESSION_KEY_TOKEN] = new_token
    st.session_state[_SESSION_KEY_EXPIRES_AT] = exp_ts or (now + 60)  # fallback prudent

    return new_token


def get_auth_headers() -> dict:
    """
    Helper prêt à l'emploi pour tout appel requests vers l'API :

        headers = get_auth_headers()
        requests.post(f"{API_BASE_URL}/triage", json=payload, headers=headers)

    Renvoie {} et affiche une erreur Streamlit si le jeton n'a pas
    pu être obtenu, plutôt que de lever une exception qui ferait
    planter toute la page.
    """
    try:
        token = get_valid_token()
        return {"Authorization": f"Bearer {token}"}
    except AuthError as exc:
        st.error(f"🔒 Authentification impossible auprès de l'API : {exc}")
        return {}


def is_authenticated() -> bool:
    """Vérifie la présence d'un jeton valide sans en forcer l'obtention immédiate."""
    try:
        get_valid_token()
        return True
    except AuthError:
        return False
