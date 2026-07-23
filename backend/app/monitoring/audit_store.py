# medical-triage-agent-ai-poc/backend/app/monitoring/audit_store.py

"""
Stockage persistant de la traçabilité des
interactions API.

Introduit à l'étape 3 pour répondre aux points 3
et 13 du cahier des charges :

    "Réaliser [...] des audits de traçabilité des
    interactions."
    "Documenter clairement les limites d'usage
    pour les utilisateurs."

Ce module est distinct de
backend/app/anonymization/audit_logger.py, qui
concerne la conformité RGPD (détection/anonymisation
PII). audit_store.py trace le trafic API lui-même
(requêtes/réponses), indépendamment de tout contenu
médical.

Format : JSON Lines (un enregistrement par ligne),
append-only, thread-safe. Choix motivé par :
- simplicité (aucune dépendance base de données
  supplémentaire, cohérent avec le périmètre POC) ;
- compatibilité directe avec les outils d'analyse
  en ligne de commande et les tests automatisés ;
- lecture paginée triviale (dernière ligne = plus
  récente).

Limite d'usage connue (à documenter pour les
utilisateurs, cf. point 13) : ce stockage est local
au conteneur et non répliqué. En environnement
Hugging Face Space, il est donc réinitialisé à
chaque redémarrage/redeploiement du Space. Il ne
doit pas être utilisé comme unique source de vérité
pour un audit réglementaire long terme.
"""

from __future__ import annotations

import json
import threading
from pathlib import Path
from typing import Any
from typing import Dict
from typing import List
from typing import Optional

_LOG_DIR = Path("backend/logs")
_LOG_FILE = _LOG_DIR / "api_audit_trail.jsonl"

_write_lock = threading.Lock()

# Journal clinique — DISTINCT du journal HTTP générique ci-dessus.
#
# Pourquoi un fichier séparé plutôt que de réutiliser record_entry()
# pour tout écrire au même endroit : l'endpoint GET /audit/ existant
# lit api_audit_trail.jsonl et mappe chaque ligne vers AuditLogEntry
# (schéma HTTP générique : request_id/path/method/status_code/...).
# Y injecter des entrées cliniques (symptômes, priorité, sortie
# brute du modèle, requires_human_review...) casserait silencieusement
# ce mapping ou, pire, exposerait des données cliniques via un
# endpoint pensé pour du monitoring technique. Un fichier + des
# fonctions dédiées gardent les deux préoccupations étanches, tout en
# réutilisant exactement le même mécanisme (JSON Lines, append-only,
# thread-safe) pour la cohérence.
#
# Corrélation avec le journal HTTP générique : chaque entrée
# clinique doit porter le même "request_id" que l'entrée HTTP
# correspondante (cf. AuditLoggingMiddleware, qui expose désormais
# ce request_id via request.state pour que triage.py puisse le
# réutiliser).
#
# Limite d'usage IDENTIQUE à celle documentée plus haut pour
# api_audit_trail.jsonl : stockage local au conteneur, non répliqué,
# réinitialisé à chaque redémarrage du Space. Pour un audit médical
# réglementaire à long terme, ce fichier ne doit pas être la seule
# source de vérité — à coordonner avec
# backend/app/anonymization/audit_logger.py (conformité RGPD) et,
# idéalement, un stockage externe persistant si ce POC évolue vers
# de la production.
_CLINICAL_LOG_DIR = Path("backend/logs")
_CLINICAL_LOG_FILE = _CLINICAL_LOG_DIR / "clinical_audit_trail.jsonl"

_clinical_write_lock = threading.Lock()


def _ensure_clinical_log_file() -> None:

    _CLINICAL_LOG_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    _CLINICAL_LOG_FILE.touch(
        exist_ok=True,
    )


def record_clinical_entry(entry: Dict[str, Any]) -> None:
    """
    Ajoute une entrée de traçabilité CLINIQUE au journal dédié
    (append-only, thread-safe). Distinct de record_entry() (journal
    HTTP générique) — voir le commentaire au-dessus de
    _CLINICAL_LOG_FILE pour la justification.

    Ne doit jamais lever d'exception vers l'appelant (route API) :
    une défaillance de l'audit ne doit jamais faire échouer une
    réponse de triage déjà calculée.
    """

    try:

        _ensure_clinical_log_file()

        line = json.dumps(
            entry,
            ensure_ascii=False,
        )

        with _clinical_write_lock:

            with open(
                _CLINICAL_LOG_FILE,
                "a",
                encoding="utf-8",
            ) as handle:

                handle.write(line + "\n")

    except Exception:
        # Volontairement silencieux : cf. docstring.
        pass


def read_clinical_entries(
    limit: Optional[int] = 100,
) -> List[Dict[str, Any]]:
    """
    Retourne les dernières entrées cliniques, de la plus récente à
    la plus ancienne. Symétrique de read_entries() ci-dessus, sur le
    journal clinique dédié.
    """

    _ensure_clinical_log_file()

    with _clinical_write_lock:

        with open(
            _CLINICAL_LOG_FILE,
            "r",
            encoding="utf-8",
        ) as handle:

            lines = handle.readlines()

    entries: List[Dict[str, Any]] = []

    for line in reversed(lines):

        line = line.strip()

        if not line:
            continue

        try:
            entries.append(json.loads(line))
        except json.JSONDecodeError:
            continue

        if limit is not None and len(entries) >= limit:
            break

    return entries


def count_clinical_entries() -> int:
    """
    Retourne le nombre total d'entrées cliniques journalisées.
    """

    _ensure_clinical_log_file()

    with _clinical_write_lock:

        with open(
            _CLINICAL_LOG_FILE,
            "r",
            encoding="utf-8",
        ) as handle:

            return sum(1 for line in handle if line.strip())


def _ensure_log_file() -> None:

    _LOG_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    _LOG_FILE.touch(
        exist_ok=True,
    )


def record_entry(entry: Dict[str, Any]) -> None:
    """
    Ajoute une entrée de traçabilité au stockage
    persistant (append-only, thread-safe).

    Ne doit jamais lever d'exception vers
    l'appelant (middleware HTTP) : une défaillance
    de l'audit ne doit pas casser une requête
    utilisateur.
    """

    try:

        _ensure_log_file()

        line = json.dumps(
            entry,
            ensure_ascii=False,
        )

        with _write_lock:

            with open(
                _LOG_FILE,
                "a",
                encoding="utf-8",
            ) as handle:

                handle.write(line + "\n")

    except Exception:
        # Volontairement silencieux : cf. docstring.
        # Un logger applicatif classique (stdout)
        # continue de capter l'information via
        # AuditLoggingMiddleware.
        pass


def read_entries(
    limit: Optional[int] = 100,
) -> List[Dict[str, Any]]:
    """
    Retourne les dernières entrées de traçabilité,
    de la plus récente à la plus ancienne.

    Parameters
    ----------
    limit:
        Nombre maximal d'entrées retournées.
        None pour retourner l'intégralité du
        journal (à utiliser avec prudence).
    """

    _ensure_log_file()

    with _write_lock:

        with open(
            _LOG_FILE,
            "r",
            encoding="utf-8",
        ) as handle:

            lines = handle.readlines()

    entries: List[Dict[str, Any]] = []

    for line in reversed(lines):

        line = line.strip()

        if not line:
            continue

        try:
            entries.append(json.loads(line))
        except json.JSONDecodeError:
            continue

        if limit is not None and len(entries) >= limit:
            break

    return entries


def count_entries() -> int:
    """
    Retourne le nombre total d'entrées journalisées.
    """

    _ensure_log_file()

    with _write_lock:

        with open(
            _LOG_FILE,
            "r",
            encoding="utf-8",
        ) as handle:

            return sum(1 for line in handle if line.strip())


def clear() -> None:
    """
    Réinitialise le journal de traçabilité.

    Réservé aux tests automatisés (test_audit_trail.py) ;
    ne pas exposer via une route API publique.
    """

    _ensure_log_file()

    with _write_lock:

        with open(
            _LOG_FILE,
            "w",
            encoding="utf-8",
        ):
            pass


def clear_clinical() -> None:
    """
    Réinitialise le journal clinique. Symétrique de clear()
    ci-dessus, sur le journal clinique dédié.

    Réservé aux tests automatisés ; ne pas exposer via une route API
    publique.
    """

    _ensure_clinical_log_file()

    with _clinical_write_lock:

        with open(
            _CLINICAL_LOG_FILE,
            "w",
            encoding="utf-8",
        ):
            pass
