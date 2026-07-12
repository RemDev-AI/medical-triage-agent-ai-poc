# medical-triage-agent-ai-poc/backend/app/core/config.py

from pydantic_settings import BaseSettings, SettingsConfigDict
from typing import List


class Settings(BaseSettings):

    APP_NAME: str = "Medical Triage AI"

    API_V1_PREFIX: str = "/api"

    # Pas de valeur par défaut : une clé faible et connue de tous
    # (ex. "CHANGE_THIS_SECRET_IN_PRODUCTION") compromettrait la
    # signature des JWT en production comme en CI. La variable doit
    # être fournie via l'environnement (.env en local, secret GitHub
    # en CI/CD).
    SECRET_KEY: str

    JWT_ALGORITHM: str = "HS256"

    ACCESS_TOKEN_EXPIRE_MINUTES: int = 60

    ALLOWED_ORIGINS: List[str] = ["http://localhost:8501", "http://localhost:3000"]

    RATE_LIMIT_REQUESTS: int = 100

    RATE_LIMIT_WINDOW_SECONDS: int = 60

    ENABLE_AUDIT_LOGGING: bool = True

    model_config = SettingsConfigDict(env_file=".env", case_sensitive=True)


settings = Settings()
