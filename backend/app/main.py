# medical-triage-agent-ai-poc/backend/app/main.py

from __future__ import annotations

from fastapi import FastAPI, Depends

from backend.app.api.router import api_router

from backend.app.api.middleware.auth_middleware import (
    JWTAuthMiddleware,
)

from backend.app.api.middleware.logging_middleware import (
    AuditLoggingMiddleware,
)

from backend.app.api.middleware.security_middleware import (
    setup_cors,
)

from backend.app.core.rate_limiter import (
    rate_limit,
)

from backend.app.deployment.huggingface.hf_space_runtime import (
    runtime_config,
)


# =========================================================
# RUNTIME DETECTION
# =========================================================

IS_HF_SPACE = runtime_config.hf_space

ENVIRONMENT = (
    "huggingface-space"
    if IS_HF_SPACE
    else "local"
)


# =========================================================
# APPLICATION
# =========================================================

app = FastAPI(
    title="Medical Triage AI API",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
)


# =========================================================
# SECURITY
# =========================================================

setup_cors(app)

app.add_middleware(
    AuditLoggingMiddleware
)

app.add_middleware(
    JWTAuthMiddleware
)


# =========================================================
# ROUTERS
#
# NOTE (correctif étape 3) :
# monitoring_router est déjà agrégé dans api_router
# (cf. backend/app/api/router.py). Il ne doit PAS être
# inclus une seconde fois ici, sous peine de dupliquer
# l'ensemble des routes /monitoring/* sous deux chemins
# distincts.
# =========================================================

app.include_router(
    api_router,
    dependencies=[
        Depends(rate_limit)
    ],
)


# =========================================================
# STARTUP EVENTS
# =========================================================

@app.on_event("startup")
async def startup_event() -> None:

    print("=" * 60)
    print("Medical Triage AI API")
    print("=" * 60)

    if IS_HF_SPACE:

        print(
            "[DEPLOYMENT] Hugging Face Space detected"
        )

        print(
            f"[MODEL] Repository : "
            f"{runtime_config.model_repository}"
        )

        print(
            f"[DEVICE] {runtime_config.device}"
        )

        print(
            f"[VLLM] Enabled : "
            f"{runtime_config.use_vllm}"
        )

        print(
            f"[4BIT] Enabled : "
            f"{runtime_config.load_in_4bit}"
        )

        print(
            f"[8BIT] Enabled : "
            f"{runtime_config.load_in_8bit}"
        )

        print(
            f"[MONITORING] Enabled : "
            f"{runtime_config.monitoring_enabled}"
        )

        print(
            f"[REQUEST_TRACKING] Enabled : "
            f"{runtime_config.request_tracking_enabled}"
        )

    else:

        print(
            "[DEPLOYMENT] Local environment"
        )

    print("=" * 60)


# =========================================================
# HEALTHCHECK
# =========================================================

@app.get(
    "/health",
    tags=["System"],
)
async def health() -> dict:

    return {
        "status": "healthy",
        "service": "Medical Triage AI",
        "version": app.version,
    }


# =========================================================
# ROOT ENDPOINT
# =========================================================

@app.get(
    "/",
    tags=["System"],
)
async def root() -> dict:

    return {
        "service": "Medical Triage AI",
        "status": "running",
        "version": app.version,
        "environment": ENVIRONMENT,
        "model_repository": (
            runtime_config.model_repository
            if IS_HF_SPACE
            else None
        ),
    }


# =========================================================
# SYSTEM INFO
# =========================================================

@app.get(
    "/system/info",
    tags=["System"],
)
async def system_info() -> dict:

    return {
        "service": "Medical Triage AI",
        "environment": ENVIRONMENT,
        "model_repository": (
            runtime_config.model_repository
        ),
        "device": (
            runtime_config.device
        ),
        "vllm_enabled": (
            runtime_config.use_vllm
        ),
        "monitoring_enabled": (
            runtime_config.monitoring_enabled
        ),
        "request_tracking_enabled": (
            runtime_config.request_tracking_enabled
        ),
        "max_input_tokens": (
            runtime_config.max_input_tokens
        ),
        "max_output_tokens": (
            runtime_config.max_output_tokens
        ),
    }
