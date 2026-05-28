# medical-triage-agent-ai-poc/frontend/streamlit_app/config/settings.py

import os

APP_TITLE = "Medical AI Triage"
PAGE_ICON = "🏥"
LAYOUT = "wide"

API_BASE_URL = os.getenv(
    "API_BASE_URL",
    "http://localhost:8000",
)

REQUEST_TIMEOUT = 30
