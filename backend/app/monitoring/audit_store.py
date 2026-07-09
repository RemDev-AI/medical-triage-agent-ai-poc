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
            entries.append(
                json.loads(line)
            )
        except json.JSONDecodeError:
            continue

        if (
            limit is not None
            and len(entries) >= limit
        ):
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

            return sum(
                1
                for line in handle
                if line.strip()
            )


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
