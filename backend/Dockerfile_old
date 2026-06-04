# medical-triage-agent-ai-poc/backend/Dockerfile

# ==========================================
# BASE IMAGE
# ==========================================
FROM python:3.11-slim

# ==========================================
# ENV VARIABLES
# ==========================================
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# ==========================================
# WORKDIR
# ==========================================
WORKDIR /app

# ==========================================
# SYSTEM DEPENDENCIES
# ==========================================
RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    git \
    && rm -rf /var/lib/apt/lists/*

# ==========================================
# COPY REQUIREMENTS
# ==========================================
COPY requirements.txt .

# ==========================================
# INSTALL PYTHON DEPENDENCIES
# ==========================================
RUN pip install --upgrade pip

RUN pip install -r requirements.txt

# ==========================================
# COPY PROJECT
# ==========================================
COPY . .

# ==========================================
# EXPOSE PORT
# ==========================================
EXPOSE 8000

# ==========================================
# DEFAULT COMMAND
# ==========================================
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
