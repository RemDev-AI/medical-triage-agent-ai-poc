# medical-triage-agent-ai-poc/backend/app/api/middleware/security_middleware.py

from fastapi.middleware.cors import CORSMiddleware

from backend.app.core.config import settings


def setup_cors(app):

    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.ALLOWED_ORIGINS,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )
