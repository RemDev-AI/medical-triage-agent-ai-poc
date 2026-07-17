# medical-triage-agent-ai-poc/frontend/streamlit_app/config/settings.py

from __future__ import annotations

import os

from streamlit_app.config.hf_space_config import (
    ui_config,
)

# =========================================================
# APPLICATION
# =========================================================

APP_TITLE = "Medical AI Triage"

PAGE_ICON = "🏥"

LAYOUT = "wide"

# =========================================================
# API
# =========================================================
#
# En environnement Hugging Face Space :
#   API_BASE_URL provient de hf_space_config.py
#
# En développement local :
#   fallback automatique vers localhost
#
# =========================================================

API_BASE_URL = ui_config.api_base_url

# =========================================================
# NETWORK
# =========================================================

REQUEST_TIMEOUT = ui_config.request_timeout

# =========================================================
# ENVIRONMENT
# =========================================================

ENVIRONMENT = ui_config.environment

MONITORING_ENABLED = ui_config.monitoring_enabled

# =========================================================
# AUTHENTIFICATION (JWT via POST /auth/token)
#
# Utilisées par streamlit_app/services/auth_client.py pour obtenir
# et renouveler automatiquement le jeton d'accès à l'API.
# =========================================================

API_ACCESS_KEY = ui_config.api_access_key

STREAMLIT_CLIENT_ID = ui_config.client_id

# =========================================================
# LEGACY FALLBACKS
# =========================================================
#
# Compatibilité avec les composants développés
# dans les phases précédentes.
#
# =========================================================

LOCAL_API_URL = os.getenv(
    "API_BASE_URL",
    "http://localhost:8000",
)

IS_PRODUCTION = ENVIRONMENT.lower() == "production"

IS_HUGGINGFACE_SPACE = (
    os.getenv(
        "HF_SPACE",
        "false",
    ).lower()
    == "true"
)
