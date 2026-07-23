# medical-triage-agent-ai-poc/backend/app/api/routes/triage.py

from __future__ import annotations

import time
import unicodedata
from datetime import datetime, timezone

from fastapi import APIRouter
from fastapi import Depends
from fastapi import HTTPException
from fastapi import Request

from app.api.schemas import (
    TriageRequest,
    TriageResponse,
)

from app.api.dependencies.inference import (
    get_triage_engine,
)

from app.llm.inference.triage_engine import TriageEngine

from app.monitoring.alerting import (
    alert_manager,
)
from app.monitoring.audit_store import (
    record_clinical_entry,
)

router = APIRouter(
    prefix="/triage",
    tags=["Triage"],
)


CONFIDENTIAL_MARKERS: tuple[str, ...] = (
    "confidential",
    "medical records",
    "medical record",
    "patient_id",
    "ssn",
    "social security",
)


_PRIORITY_MAPPING: dict[str, str] = {
    "faible": "low",
    "low": "low",
    "modere": "medium",
    "moyen": "medium",
    "medium": "medium",
    "eleve": "high",
    "high": "high",
    "urgent": "urgent",
    "critique": "urgent",
}


def _normalize_priority(raw_value: str) -> str:
    """
    TriageEngine.parse_response() renvoie "priority" en français
    (ex: "FAIBLE", "ÉLEVÉ"), alors que TriageResponse.priority_level
    doit appartenir au domaine anglais {low, medium, high, urgent}
    attendu par les contrats d'API.

    On normalise la casse et les accents avant de mapper vers la
    valeur canonique. Si la valeur est vide ou inconnue, on retombe
    sur "medium" par prudence et on lève une alerte pour tracer les
    sorties inattendues du modèle plutôt que de les ignorer
    silencieusement.
    """

    if not raw_value:
        # SÉCURITÉ CLINIQUE (2026-07-21, cf. cahier des charges CHSA) :
        # une valeur de priorité absente/vide ne doit jamais retomber
        # sur "medium" — valeur médiane silencieuse qui peut masquer
        # un cas critique. On escalade vers "urgent", cohérent avec
        # TriageEngine.normalize_priority qui applique le même
        # principe de précaution côté moteur (repli vers CRITIQUE).
        return "urgent"

    normalized = unicodedata.normalize("NFKD", raw_value.strip().lower())
    normalized = "".join(c for c in normalized if not unicodedata.combining(c))

    if normalized not in _PRIORITY_MAPPING:
        try:
            alert_manager.raise_alert(
                code="TRIAGE_UNKNOWN_PRIORITY",
                message=f"Unknown priority value returned by triage engine: {raw_value!r}",
            )
        except Exception:
            pass
        # Idem : repli vers "urgent", jamais "medium", en cas de
        # valeur de priorité non reconnue.
        return "urgent"

    return _PRIORITY_MAPPING[normalized]


def _strip_confidential_leakage(text: str) -> str:

    sanitized = text

    for marker in CONFIDENTIAL_MARKERS:
        if marker.lower() in sanitized.lower():
            lowered = sanitized.lower()
            marker_lower = marker.lower()
            result = []
            idx = 0
            while True:
                pos = lowered.find(marker_lower, idx)
                if pos == -1:
                    result.append(sanitized[idx:])
                    break
                result.append(sanitized[idx:pos])
                result.append("[redacted]")
                idx = pos + len(marker_lower)
            sanitized = "".join(result)
            lowered = sanitized.lower()

    return sanitized


def _split_recommendations(raw: str) -> list[str]:
    """
    TriageEngine.parse_response() renvoie "recommendations" comme un
    unique bloc de texte (issu de l'extraction de la section
    "RECOMMANDATIONS:" dans la sortie du modèle), alors que
    TriageResponse.recommendations attend une List[str].

    On segmente ligne par ligne, en retirant les puces ("-", "*")
    éventuelles. Si aucune ligne exploitable n'est trouvée mais que le
    modèle a bien renvoyé du texte, on renvoie ce texte comme
    recommandation unique plutôt que de perdre l'information.
    """

    if not raw or raw == "Non disponible":
        return []

    lines = [
        line.strip().lstrip("-*").strip() for line in raw.splitlines() if line.strip()
    ]

    return lines if lines else [raw.strip()]


@router.post(
    "/",
    response_model=TriageResponse,
)
async def triage_route(
    payload: TriageRequest,
    request: Request,
    triage_engine: TriageEngine = Depends(
        get_triage_engine,
    ),
):
    start_time = time.perf_counter()

    # Corrélation avec le journal HTTP générique (cf.
    # AuditLoggingMiddleware, qui expose désormais ce request_id via
    # request.state). Repli sur "unknown" si jamais ce middleware
    # n'était pas monté dans un contexte donné (ex: certains tests
    # unitaires qui appellent la route sans passer par la stack
    # middleware complète), pour ne jamais faire échouer le triage à
    # cause de la traçabilité.
    request_id = getattr(request.state, "request_id", "unknown")

    try:

        # ------------------------------------------------------
        # Adaptation de types : TriageRequest expose des champs
        # texte libre (str), alors que TriageEngine.run_triage
        # attend des listes (symptoms: List[str],
        # medical_history: Optional[List[str]]).
        #
        # Hypothèse de migration : on transmet chaque champ texte
        # comme un unique élément de liste plutôt que de le
        # découper arbitrairement (par virgule, etc.), afin de ne
        # pas altérer le contenu clinique saisi par l'utilisateur.
        # A confirmer si un découpage plus fin est souhaité.
        # ------------------------------------------------------

        result = await triage_engine.run_triage(
            patient_age=payload.age if payload.age is not None else 0,
            symptoms=[payload.symptoms],
            medical_history=(
                [payload.medical_history] if payload.medical_history else None
            ),
        )

    except Exception:

        try:
            alert_manager.raise_alert(
                code="TRIAGE_ERROR",
                message="Unexpected error during triage processing.",
            )
        except Exception:
            pass

        # Log clinique même en cas d'échec : un audit doit pouvoir
        # voir qu'un patient a soumis une requête qui n'a JAMAIS
        # obtenu de triage, pas seulement les requêtes réussies.
        record_clinical_entry(
            {
                "request_id": request_id,
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "patient_id": payload.patient_id,
                "age": payload.age,
                "symptoms": payload.symptoms,
                "medical_history": payload.medical_history,
                "priority_context": payload.priority_context,
                "error": True,
                "priority_level": None,
                "confidence_score": None,
                "requires_human_review": True,
            }
        )

        raise HTTPException(
            status_code=500,
            detail="An unexpected error occurred while processing the triage request.",
        )

    latency_seconds = time.perf_counter() - start_time

    latency_ms = latency_seconds * 1000

    try:
        alert_manager.evaluate_latency(latency_ms)
    except Exception:
        pass

    triage_data = result.get("triage", {})

    justification_raw = triage_data.get(
        "justification",
        "",
    )

    recommendations_raw = triage_data.get(
        "recommendations",
        "",
    )

    normalized_priority_level = _normalize_priority(
        triage_data.get(
            "priority",
            "UNKNOWN",
        ),
    )

    raw_priority = triage_data.get("priority", "UNKNOWN")

    # requires_human_review vient normalement de TriageEngine
    # (cf. parse_response), mais on applique un défaut prudent à
    # True si le champ est absent (ex: ancienne version du moteur
    # qui ne le renvoie pas encore) plutôt qu'un défaut à False, qui
    # masquerait silencieusement le besoin de revue.
    requires_human_review = triage_data.get(
        "requires_human_review",
        True,
    )

    # Garde-fou supplémentaire, indépendant du moteur : si le
    # mapping FR -> EN de la priorité a lui-même dû recourir à son
    # propre fallback ("urgent" par défaut, cf. _normalize_priority
    # ci-dessus), la revue humaine est requise ici aussi, même si le
    # moteur avait — à tort ou à raison — jugé sa propre extraction
    # suffisamment fiable.
    if raw_priority:
        _normalized_check = unicodedata.normalize(
            "NFKD", str(raw_priority).strip().lower()
        )
        _normalized_check = "".join(
            c for c in _normalized_check if not unicodedata.combining(c)
        )
        if _normalized_check not in _PRIORITY_MAPPING:
            requires_human_review = True
    else:
        requires_human_review = True

    triage_response = TriageResponse(
        priority_level=normalized_priority_level,
        justification=_strip_confidential_leakage(justification_raw),
        recommendations=[
            _strip_confidential_leakage(item)
            for item in _split_recommendations(recommendations_raw)
        ],
        confidence_score=triage_data.get(
            "confidence_score",
            0.0,
        ),
        requires_human_review=requires_human_review,
        generated_at=time.strftime("%Y-%m-%dT%H:%M:%S"),
        latency_seconds=round(
            latency_seconds,
            3,
        ),
    )

    # Log clinique complet, corrélé au request_id du journal HTTP
    # générique (cf. AuditLoggingMiddleware). Contient volontairement
    # la sortie BRUTE du modèle (raw_response) en plus des champs
    # parsés : en cas de désaccord clinique lors d'un audit, il faut
    # pouvoir vérifier si l'erreur vient du modèle ou du parsing.
    #
    # NOTE : ce journal contient des données patient. Cf. limites
    # d'usage documentées dans audit_store.py (stockage local, non
    # répliqué) — à coordonner avec la politique RGPD existante
    # (app/anonymization/audit_logger.py) avant tout déploiement
    # au-delà du POC.
    record_clinical_entry(
        {
            "request_id": request_id,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "patient_id": payload.patient_id,
            "age": payload.age,
            "symptoms": payload.symptoms,
            "medical_history": payload.medical_history,
            "priority_context": payload.priority_context,
            "model_name": getattr(triage_engine, "model_name", "unknown"),
            "raw_priority": raw_priority,
            "priority_level": normalized_priority_level,
            "justification": triage_response.justification,
            "recommendations": triage_response.recommendations,
            "confidence_score": triage_response.confidence_score,
            "requires_human_review": triage_response.requires_human_review,
            "raw_response": result.get("raw_response"),
            "latency_seconds": triage_response.latency_seconds,
            "error": False,
        }
    )

    return triage_response
