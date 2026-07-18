# medical-triage-agent-ai-poc/frontend/Dockerfile

# =========================================================
# FRONTEND DOCKERFILE
# POC AGENT IA TRIAGE MÉDICAL
# =========================================================

# ---------------------------------------------------------
# IMAGE PYTHON
# ---------------------------------------------------------
FROM python:3.14-slim

# ---------------------------------------------------------
# VARIABLES D’ENVIRONNEMENT
# ---------------------------------------------------------
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV PIP_NO_CACHE_DIR=1
ENV STREAMLIT_SERVER_PORT=8501
ENV STREAMLIT_SERVER_ADDRESS=0.0.0.0

# ---------------------------------------------------------
# DOSSIER APPLICATION
# ---------------------------------------------------------
WORKDIR /app

# ---------------------------------------------------------
# INSTALLATION DÉPENDANCES SYSTÈME
# ---------------------------------------------------------
RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    git \
    && rm -rf /var/lib/apt/lists/*

# ---------------------------------------------------------
# INSTALLATION DÉPENDANCES PYTHON
# ---------------------------------------------------------
COPY requirements.txt .

RUN pip install --upgrade pip && \
    pip install -r requirements.txt

# ---------------------------------------------------------
# COPIE DU PROJET FRONTEND
# ---------------------------------------------------------
COPY . .

# ---------------------------------------------------------
# PORT STREAMLIT
# ---------------------------------------------------------
EXPOSE 8501

# ---------------------------------------------------------
# HEALTHCHECK
# ---------------------------------------------------------
HEALTHCHECK CMD curl --fail http://localhost:8501/_stcore/health || exit 1

# ---------------------------------------------------------
# LANCEMENT STREAMLIT
# ---------------------------------------------------------
CMD ["streamlit", "run", "streamlit_app/app.py", "--server.port=8501", "--server.address=0.0.0.0"]
