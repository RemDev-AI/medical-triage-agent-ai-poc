# medical-triage-agent-ai-poc/backend/app/api/middleware/security_middleware.py

from fastapi.middleware.cors import CORSMiddleware

from app.core.config import settings


def setup_cors(app):
    """
    Configure CORS à partir de settings.ALLOWED_ORIGINS (surchargeable
    via variable d'environnement, cf. app/core/config.py).

    Couvre par défaut :
      - http://localhost:8501            (frontend/Dockerfile, dev local)
      - http://localhost:3000            (dev local, autre frontend éventuel)
      - Space UI Hugging Face             (frontend/Dockerfile.hf, prod)

    allow_credentials=True impose que allow_origins liste des origines
    explicites (jamais "*"), ce qui est déjà le cas ici : le paramètre
    ALLOWED_ORIGINS reste une liste précise plutôt qu'un joker.
    """

    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.ALLOWED_ORIGINS,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )
