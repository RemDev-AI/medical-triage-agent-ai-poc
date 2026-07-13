# ==========================================================
# medical-triage-agent-ai-poc/backend/Dockerfile
# Backend FastAPI
# Google Colab GPU Free compatible
#
# IMPORTANT : ce Dockerfile doit être buildé avec la RACINE
# du dépôt comme contexte (et non ./backend), afin de
# préserver la structure de package "backend.app.*" utilisée
# par tous les imports Python du projet.
#
# Exemple :
#   docker build -f backend/Dockerfile -t medical-triage-backend .
#
# Cf. docker-compose.yml (context: ., dockerfile: backend/Dockerfile)
# ==========================================================

FROM python:3.14-slim

# ==========================================================
# ENVIRONMENT
# ==========================================================

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

ENV APP_HOME=/app

WORKDIR ${APP_HOME}

# ==========================================================
# SYSTEM DEPENDENCIES
# ==========================================================

RUN apt-get update && apt-get install -y \
    build-essential \
    gcc \
    g++ \
    curl \
    git \
    && rm -rf /var/lib/apt/lists/*

# ==========================================================
# PYTHON DEPENDENCIES
# ==========================================================

COPY requirements.txt ./requirements.txt
RUN pip install --upgrade pip
RUN pip install --no-cache-dir -r requirements.txt

# ==========================================================
# APPLICATION
# ==========================================================

# Copie du package "backend" complet (préserve backend.app.*)
COPY . ./  

# ==========================================================
# NETWORK
# ==========================================================

EXPOSE 8000

# ==========================================================
# HEALTHCHECK
# ==========================================================

HEALTHCHECK --interval=30s \
            --timeout=10s \
            --start-period=20s \
            --retries=5 \
            CMD curl -f http://localhost:8000/health || exit 1

# ==========================================================
# STARTUP
# ==========================================================

CMD ["uvicorn", "backend.app.main:app", "--host", "0.0.0.0", "--port", "8000"]
