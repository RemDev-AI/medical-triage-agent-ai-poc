# medical-triage-agent-ai-poc/backend/app/core/config.py

from pydantic_settings import BaseSettings
from typing import List


class Settings(BaseSettings):

    APP_NAME: str = "Medical Triage AI"

    API_V1_PREFIX: str = "/api"

    SECRET_KEY: str = "CHANGE_THIS_SECRET_IN_PRODUCTION"

    JWT_ALGORITHM: str = "HS256"

    ACCESS_TOKEN_EXPIRE_MINUTES: int = 60

    ALLOWED_ORIGINS: List[str] = ["http://localhost:8501", "http://localhost:3000"]

    RATE_LIMIT_REQUESTS: int = 100

    RATE_LIMIT_WINDOW_SECONDS: int = 60

    ENABLE_AUDIT_LOGGING: bool = True

    class Config:
        env_file = ".env"
        case_sensitive = True


settings = Settings()
