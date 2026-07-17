# medical-triage-agent-ai-poc/backend/app/core/config.py

from pydantic_settings import BaseSettings, SettingsConfigDict
from pydantic import field_validator
from typing import List, Union


class Settings(BaseSettings):

    APP_NAME: str = "Medical Triage AI"

    API_V1_PREFIX: str = "/api"

    # Pas de valeur par défaut : une clé faible et connue de tous
    # (ex. "CHANGE_THIS_SECRET_IN_PRODUCTION") compromettrait la
    # signature des JWT en production comme en CI. La variable doit
    # être fournie via l'environnement (.env en local, secret GitHub
    # en CI/CD, secret Hugging Face Space en déploiement).
    SECRET_KEY: str

    JWT_ALGORITHM: str = "HS256"

    ACCESS_TOKEN_EXPIRE_MINUTES: int = 60

    # ------------------------------------------------------------
    # API_ACCESS_KEY
    #
    # Secret partage, distinct de SECRET_KEY, requis par
    # POST /auth/token pour obtenir un JWT. Sans lui, n'importe quel
    # client connaissant simplement l'URL du Space pourrait generer
    # ses propres jetons valides, ce qui viderait la protection JWT
    # de son sens. A definir dans les Secrets du Space (API), au
    # meme titre que SECRET_KEY.
    # ------------------------------------------------------------
    API_ACCESS_KEY: str

    # ------------------------------------------------------------
    # ALLOWED_ORIGINS
    #
    # Valeur par défaut couvrant le développement local (Streamlit
    # sur 8501, un éventuel frontend sur 3000) ainsi que les deux
    # Hugging Face Spaces (API et UI) utilisés en production.
    #
    # Elle reste entièrement surchargeable via la variable
    # d'environnement ALLOWED_ORIGINS (cf. .env local ou Secrets HF),
    # ce qui permet d'ajouter/retirer des origines sans toucher au
    # code, par exemple si l'URL du Space change.
    #
    # Formats acceptés côté environnement :
    #   ALLOWED_ORIGINS=https://a.hf.space,https://b.hf.space
    #   ALLOWED_ORIGINS=["https://a.hf.space","https://b.hf.space"]
    # ------------------------------------------------------------
    ALLOWED_ORIGINS: List[str] = [
        "http://localhost:8501",
        "http://localhost:3000",
        "https://remdev-ai-medical-triage-agent-ai-poc-ui.hf.space",
    ]

    @field_validator("ALLOWED_ORIGINS", mode="before")
    @classmethod
    def _parse_allowed_origins(cls, value: Union[str, List[str]]):
        """
        Permet de fournir ALLOWED_ORIGINS sous forme de chaîne simple
        séparée par des virgules dans l'environnement (cas le plus
        pratique pour les Secrets Hugging Face / GitHub Actions), en
        plus du format liste JSON déjà supporté nativement par
        pydantic-settings.
        """
        if isinstance(value, str):
            stripped = value.strip()

            # Support du format JSON classique : ["a", "b"]
            if stripped.startswith("["):
                return value

            # Format "origine1,origine2,origine3"
            return [origin.strip() for origin in stripped.split(",") if origin.strip()]

        return value

    RATE_LIMIT_REQUESTS: int = 100

    RATE_LIMIT_WINDOW_SECONDS: int = 60

    ENABLE_AUDIT_LOGGING: bool = True

    model_config = SettingsConfigDict(env_file=".env", case_sensitive=True)


settings = Settings()
