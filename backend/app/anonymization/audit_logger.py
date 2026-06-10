# medical-triage-agent-ai-poc/backend/app/anonymization/audit_logger.py

"""
Logging d'audit RGPD.

Centralise la journalisation des opérations :

- Détection PII
- Anonymisation
- Validation
- Contrôles de conformité

Compatible :
- FastAPI
- Modal
- GitHub Actions
- Pytest
"""

from __future__ import annotations

import logging
from pathlib import Path

# ==========================================================
# LOG DIRECTORY
# ==========================================================

LOG_DIR = Path("backend/logs")
LOG_DIR.mkdir(
    parents=True,
    exist_ok=True,
)

LOG_FILE = LOG_DIR / "anonymization_audit.log"

# ==========================================================
# LOGGER CONFIGURATION
# ==========================================================

LOGGER_NAME = "rgpd_audit"

audit_logger = logging.getLogger(
    LOGGER_NAME
)

audit_logger.setLevel(
    logging.INFO
)

# Évite l'ajout multiple de handlers
if not audit_logger.handlers:

    formatter = logging.Formatter(
        "%(asctime)s | %(levelname)s | %(message)s"
    )

    file_handler = logging.FileHandler(
        LOG_FILE,
        encoding="utf-8",
    )

    file_handler.setFormatter(
        formatter
    )

    audit_logger.addHandler(
        file_handler
    )

    # Évite la duplication via le logger racine
    audit_logger.propagate = False

# ==========================================================
# LOCAL TEST
# ==========================================================

if __name__ == "__main__":

    audit_logger.info(
        "Audit logger initialized"
    )

    audit_logger.info(
        "PII detection executed | "
        "language=fr | "
        "findings=3"
    )

    audit_logger.info(
        "Anonymization applied | "
        "language=en | "
        "strategy=replace | "
        "findings=2"
    )

    print(
        f"Audit log file: {LOG_FILE}"
    )
