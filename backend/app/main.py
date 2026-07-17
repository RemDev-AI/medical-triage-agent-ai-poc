# medical-triage-agent-ai-poc/backend/app/main.py

from __future__ import annotations

from contextlib import asynccontextmanager

from fastapi import FastAPI, Depends
from fastapi.openapi.utils import get_openapi
from fastapi.security import HTTPBearer

from app.api.router import api_router

from app.api.middleware.auth_middleware import (
    JWTAuthMiddleware,
)

from app.api.middleware.logging_middleware import (
    AuditLoggingMiddleware,
)

from app.api.middleware.security_middleware import (
    setup_cors,
)

from app.core.rate_limiter import (
    rate_limit,
)

from app.deployment.huggingface.hf_space_runtime import (
    runtime_config,
)

# =========================================================
# RUNTIME DETECTION
# =========================================================

IS_HF_SPACE = runtime_config.hf_space

ENVIRONMENT = "huggingface-space" if IS_HF_SPACE else "local"


# =========================================================
# LIFESPAN (remplace @app.on_event("startup"), déprécié)
# =========================================================


@asynccontextmanager
async def lifespan(app: FastAPI):

    print("=" * 60)
    print("Medical Triage AI API")
    print("=" * 60)

    if IS_HF_SPACE:

        print("[DEPLOYMENT] Hugging Face Space detected")

        print(f"[MODEL] Repository : " f"{runtime_config.model_repository}")

        print(f"[DEVICE] {runtime_config.device}")

        print(f"[VLLM] Enabled : " f"{runtime_config.use_vllm}")

        print(f"[4BIT] Enabled : " f"{runtime_config.load_in_4bit}")

        print(f"[8BIT] Enabled : " f"{runtime_config.load_in_8bit}")

        print(f"[MONITORING] Enabled : " f"{runtime_config.monitoring_enabled}")

        print(
            f"[REQUEST_TRACKING] Enabled : "
            f"{runtime_config.request_tracking_enabled}"
        )

    else:

        print("[DEPLOYMENT] Local environment")

    print("=" * 60)

    # NOTE (migration InferenceClient -> inférence locale) :
    # le chargement du modèle/adaptateur n'a volontairement PAS lieu
    # ici. Il est différé (lazy singleton) jusqu'au premier appel réel
    # de get_triage_engine() / get_generation_context()
    # (cf. app/api/dependencies/inference.py), pour ne jamais impacter
    # le démarrage de l'API (health checks, /docs) ni l'exécution des
    # tests — même principe que vllm_engine.get_vllm_engine().

    yield

    # (aucune action de shutdown requise pour le moment ;
    # cet emplacement est prêt pour d'éventuels nettoyages
    # futurs — fermeture de connexions, flush de buffers, etc.)


# =========================================================
# APPLICATION
# =========================================================

app = FastAPI(
    title="Medical Triage AI API",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
    lifespan=lifespan,
)


# =========================================================
# SECURITY
# =========================================================

setup_cors(app)

app.add_middleware(AuditLoggingMiddleware)

app.add_middleware(JWTAuthMiddleware)


# ---------------------------------------------------------
# OpenAPI / Swagger : bouton "Authorize"
#
# HTTPBearer() ci-dessous ne fait AUCUNE vérification (le
# middleware JWTAuthMiddleware s'en charge toujours, en amont
# de chaque requête). Son unique rôle est de déclarer un schéma
# de sécurité "bearerAuth" dans le schéma OpenAPI généré, afin
# que Swagger UI affiche un bouton "Authorize" permettant de
# coller un JWT (obtenu via POST /auth/token) et de l'envoyer
# automatiquement sur chaque requête via
# "Authorization: Bearer <jwt>".
#
# On l'ajoute comme dépendance "fantôme" sur api_router (elle
# ne bloque jamais la requête elle-même) uniquement pour que
# FastAPI l'inclue dans le schéma OpenAPI de ces routes.
# ---------------------------------------------------------

bearer_scheme = HTTPBearer(auto_error=False)


def custom_openapi():
    if app.openapi_schema:
        return app.openapi_schema

    openapi_schema = get_openapi(
        title=app.title,
        version=app.version,
        description="Medical Triage AI API",
        routes=app.routes,
    )

    openapi_schema.setdefault("components", {}).setdefault("securitySchemes", {})[
        "bearerAuth"
    ] = {
        "type": "http",
        "scheme": "bearer",
        "bearerFormat": "JWT",
    }

    # Applique le schéma "bearerAuth" à toutes les opérations, sauf
    # celles qui n'en ont pas besoin (health, root, auth/token lui-même).
    public_paths = {"/", "/health", "/auth/token"}

    for path, methods in openapi_schema.get("paths", {}).items():
        if path in public_paths:
            continue
        for operation in methods.values():
            operation.setdefault("security", [{"bearerAuth": []}])

    app.openapi_schema = openapi_schema
    return app.openapi_schema


app.openapi = custom_openapi


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
    dependencies=[Depends(rate_limit), Depends(bearer_scheme)],
)


# =========================================================
# HEALTHCHECK
# =========================================================


@app.get("/health", tags=["System"])
async def health() -> dict:
    return {
        "status": "ok",  # avant : "healthy"
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
        "model_repository": (runtime_config.model_repository if IS_HF_SPACE else None),
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
        "model_repository": (runtime_config.model_repository),
        "device": (runtime_config.device),
        "vllm_enabled": (runtime_config.use_vllm),
        "monitoring_enabled": (runtime_config.monitoring_enabled),
        "request_tracking_enabled": (runtime_config.request_tracking_enabled),
        "max_input_tokens": (runtime_config.max_input_tokens),
        "max_output_tokens": (runtime_config.max_output_tokens),
    }
