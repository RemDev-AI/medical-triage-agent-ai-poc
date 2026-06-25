# ==========================================================
# medical-triage-agent-ai-poc/backend/Dockerfile
# Backend FastAPI
# Google Colab GPU Free compatible
# ==========================================================

FROM python:3.11-slim

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

COPY requirements.txt .

RUN pip install --upgrade pip

RUN pip install --no-cache-dir -r requirements.txt

# ==========================================================
# APPLICATION
# ==========================================================

COPY . .

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

CMD [
    "uvicorn",
    "app.main:app",
    "--host",
    "0.0.0.0",
    "--port",
    "8000"
]
