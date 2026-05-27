# medical-triage-agent-ai-poc/backend/app/anonymization/audit_logger.py

"""
Logging RGPD auditabilité.
"""

from __future__ import annotations

import logging
from pathlib import Path

LOG_DIR = Path("backend/logs")
LOG_DIR.mkdir(parents=True, exist_ok=True)

LOG_FILE = LOG_DIR / "anonymization_audit.log"

logging.basicConfig(
    filename=str(LOG_FILE),
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
)

audit_logger = logging.getLogger("rgpd_audit")
