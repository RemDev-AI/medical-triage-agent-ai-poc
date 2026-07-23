# medical-triage-agent-ai-poc/backend/app/api/routes/audit.py

from __future__ import annotations

from fastapi import APIRouter
from fastapi import Depends
from fastapi import Query

from app.api.schemas import (
    AuditLogEntry,
    AuditResponse,
    ClinicalAuditLogEntry,
    ClinicalAuditResponse,
)
from app.monitoring.audit_store import (
    count_entries,
    count_clinical_entries,
    read_entries,
    read_clinical_entries,
)
from app.core.security import require_scope

router = APIRouter(prefix="/audit", tags=["Audit"])


@router.get(
    "/",
    response_model=AuditResponse,
    dependencies=[Depends(require_scope("audit"))],
)
async def audit_logs(
    limit: int = Query(
        default=100,
        ge=1,
        le=1000,
        description=(
            "Nombre maximal d'entrées retournées, "
            "des plus récentes aux plus anciennes."
        ),
    ),
):
    """
    Traçabilité des interactions API (correctif
    étape 3).

    Remplace les données mockées en dur par une
    lecture réelle du journal persistant
    (backend/app/monitoring/audit_store.py),
    alimenté par AuditLoggingMiddleware pour
    chaque requête HTTP traitée par l'API.

    Limite d'usage : ce journal est local au
    conteneur et non répliqué (cf. audit_store.py).
    """

    raw_entries = read_entries(limit=limit)

    logs = [
        AuditLogEntry(
            request_id=entry.get("request_id", "unknown"),
            endpoint=entry.get("path", "unknown"),
            method=entry.get("method", "unknown"),
            status_code=entry.get("status_code", 0),
            timestamp=entry.get("timestamp"),
            latency_ms=entry.get("latency_ms", 0.0),
            client_ip=entry.get("client_ip"),
        )
        for entry in raw_entries
    ]

    return AuditResponse(
        total_logs=count_entries(),
        logs=logs,
    )


@router.get(
    "/clinical",
    response_model=ClinicalAuditResponse,
    dependencies=[Depends(require_scope("audit"))],
)
async def clinical_audit_logs(
    limit: int = Query(
        default=100,
        ge=1,
        le=1000,
        description=(
            "Nombre maximal d'entrées retournées, "
            "des plus récentes aux plus anciennes."
        ),
    ),
):
    """
    Traçabilité CLINIQUE des interactions de triage (distincte de
    GET /audit/, qui ne couvre que le trafic HTTP générique).

    Répond à l'exigence CHSA de "traçabilité de chaque interaction
    pour les audits médicaux" : chaque entrée contient les données
    patient soumises, la sortie brute du modèle, le résultat parsé
    (priorité/justification/recommandations), le score de confiance
    et l'indicateur requires_human_review.

    Chaque entrée porte un request_id permettant la corrélation avec
    l'entrée correspondante de GET /audit/ (journal HTTP générique).

    ATTENTION : ce journal contient des données patient. Cet
    endpoint ne doit être exposé qu'à du personnel autorisé — à
    sécuriser (authentification/autorisation) avant tout déploiement
    au-delà du POC, et à coordonner avec la politique RGPD existante
    (app/anonymization/audit_logger.py).

    Limite d'usage : ce journal est local au conteneur et non
    répliqué (cf. audit_store.py).
    """

    raw_entries = read_clinical_entries(limit=limit)

    logs = [
        ClinicalAuditLogEntry(
            request_id=entry.get("request_id", "unknown"),
            timestamp=entry.get("timestamp"),
            patient_id=entry.get("patient_id"),
            age=entry.get("age"),
            symptoms=entry.get("symptoms"),
            medical_history=entry.get("medical_history"),
            priority_context=entry.get("priority_context"),
            model_name=entry.get("model_name"),
            raw_priority=entry.get("raw_priority"),
            priority_level=entry.get("priority_level"),
            justification=entry.get("justification"),
            recommendations=entry.get("recommendations"),
            confidence_score=entry.get("confidence_score"),
            requires_human_review=entry.get("requires_human_review"),
            raw_response=entry.get("raw_response"),
            latency_seconds=entry.get("latency_seconds"),
            error=entry.get("error", False),
        )
        for entry in raw_entries
    ]

    return ClinicalAuditResponse(
        total_logs=count_clinical_entries(),
        logs=logs,
    )
