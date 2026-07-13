# medical-triage-agent-ai-poc/backend/app/training/evaluation/evaluation_report.py

"""
Clinical evaluation report generation.

Responsibilities:
- Generate JSON report
- Generate Markdown report
- Persist evaluation artifacts

Generated files:
- evaluation_report.json
- evaluation_report.md

Compatible with:
- SFT evaluation
- DPO evaluation
- Google Colab
- HF Hub
- Weights & Biases
- clinical_eval_runner.py
"""

from __future__ import annotations

import json
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from typing import Dict
from typing import Optional

import yaml

logger = logging.getLogger(__name__)

# ============================================================
# FIX EVAL-6 — evaluation_config.yaml n'était lu nulle part ici :
# reporting.include_metadata / reporting.include_thresholds n'avaient
# donc aucun effet (le Markdown n'incluait ni seuils ni métadonnées,
# même quand ces flags étaient à true). On charge le YAML pour piloter
# réellement le contenu des rapports.
# json_filename / markdown_filename restent gérés via les paramètres
# filename= existants (déjà alignés par coïncidence avec le YAML) ;
# seuls include_metadata/include_thresholds étaient de vraies clés
# mortes.
# ============================================================
EVALUATION_CONFIG_PATH = Path(__file__).parent / "evaluation_config.yaml"


def load_evaluation_config() -> Dict[str, Any]:
    with open(EVALUATION_CONFIG_PATH, "r", encoding="utf-8") as file:
        return yaml.safe_load(file)


# ============================================================
# CONSTANTS
# ============================================================

DEFAULT_JSON_FILENAME = "evaluation_report.json"
DEFAULT_MARKDOWN_FILENAME = "evaluation_report.md"


# ============================================================
# HELPERS
# ============================================================


def _utc_timestamp() -> str:
    """
    Returns UTC timestamp in ISO-8601 format.
    """

    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def _ensure_directory(
    output_dir: str | Path,
) -> Path:
    """
    Create output directory if needed.
    """

    path = Path(output_dir)

    path.mkdir(
        parents=True,
        exist_ok=True,
    )

    return path


# ============================================================
# JSON REPORT
# ============================================================


def save_json_report(
    report_data: Dict[str, Any],
    output_dir: str | Path,
    filename: str = DEFAULT_JSON_FILENAME,
) -> Path:
    """
    Save evaluation report as JSON.

    Returns:
        Path to generated file.
    """

    output_path = _ensure_directory(output_dir) / filename

    with open(
        output_path,
        "w",
        encoding="utf-8",
    ) as fp:
        json.dump(
            report_data,
            fp,
            indent=4,
            ensure_ascii=False,
        )

    return output_path


# ============================================================
# MARKDOWN REPORT
# ============================================================


def build_markdown_report(
    report_data: Dict[str, Any],
) -> str:
    """
    Generate Markdown report.

    Expected structure:

    {
        "model_name": ...,
        "evaluation_timestamp": ...,
        "clinical_metrics": {...},
        "safety": {...},
        "overall_status": ...
    }
    """

    model_name = report_data.get(
        "model_name",
        "unknown_model",
    )

    timestamp = report_data.get(
        "evaluation_timestamp",
        _utc_timestamp(),
    )

    overall_status = report_data.get(
        "overall_status",
        "UNKNOWN",
    )

    clinical_metrics = report_data.get(
        "clinical_metrics",
        {},
    )

    safety = report_data.get(
        "safety",
        {},
    )

    # FIX EVAL-6 — seuils et métadonnées, jusqu'ici jamais rendus dans le
    # Markdown même quand include_thresholds/include_metadata=true.
    eval_config = load_evaluation_config()
    reporting_config = eval_config.get("reporting", {})

    thresholds = report_data.get("thresholds", {})
    metadata = report_data.get("metadata", {})

    lines = [
        "# Clinical Evaluation Report",
        "",
        f"**Model:** {model_name}",
        f"**Timestamp:** {timestamp}",
        f"**Status:** {overall_status}",
        "",
        "---",
        "",
        "## Clinical Metrics",
        "",
    ]

    for metric_name, metric_value in clinical_metrics.items():
        lines.append(
            f"- **{metric_name}**: " f"{metric_value:.4f}"
            if isinstance(
                metric_value,
                (float, int),
            )
            else f"- **{metric_name}**: {metric_value}"
        )

    lines.extend(
        [
            "",
            "---",
            "",
            "## Safety Evaluation",
            "",
        ]
    )

    for metric_name, metric_value in safety.items():
        if isinstance(
            metric_value,
            (float, int),
        ):
            lines.append(f"- **{metric_name}**: " f"{metric_value:.4f}")
        else:
            lines.append(f"- **{metric_name}**: " f"{metric_value}")

    # FIX EVAL-6 — section Thresholds, rendue uniquement si
    # reporting.include_thresholds=true ET des seuils sont disponibles
    # dans report_data (fournis par create_evaluation_report()).
    if reporting_config.get("include_thresholds", False) and thresholds:
        lines.extend(
            [
                "",
                "---",
                "",
                "## Thresholds",
                "",
            ]
        )
        for threshold_name, threshold_value in thresholds.items():
            lines.append(
                f"- **{threshold_name}**: " f"{threshold_value:.4f}"
                if isinstance(threshold_value, (float, int))
                else f"- **{threshold_name}**: {threshold_value}"
            )

    # FIX EVAL-6 — section Metadata, rendue uniquement si
    # reporting.include_metadata=true ET des métadonnées sont
    # disponibles.
    if reporting_config.get("include_metadata", False) and metadata:
        lines.extend(
            [
                "",
                "---",
                "",
                "## Metadata",
                "",
            ]
        )
        for meta_key, meta_value in metadata.items():
            lines.append(f"- **{meta_key}**: {meta_value}")

    lines.extend(
        [
            "",
            "---",
            "",
            "## Summary",
            "",
            (
                "Model passes clinical evaluation."
                if overall_status == "PASS"
                else "Model does not satisfy clinical " "evaluation requirements."
            ),
            "",
        ]
    )

    return "\n".join(lines)


def save_markdown_report(
    report_data: Dict[str, Any],
    output_dir: str | Path,
    filename: str = DEFAULT_MARKDOWN_FILENAME,
) -> Path:
    """
    Save Markdown report.

    Returns:
        Path to generated file.
    """

    output_path = _ensure_directory(output_dir) / filename

    markdown_content = build_markdown_report(report_data)

    with open(
        output_path,
        "w",
        encoding="utf-8",
    ) as fp:
        fp.write(markdown_content)

    return output_path


# ============================================================
# COMPLETE REPORT
# ============================================================


def create_evaluation_report(
    *,
    model_name: str,
    clinical_metrics: Dict[str, Any],
    safety: Dict[str, Any],
    overall_status: str,
    metadata: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """
    Build canonical report structure.
    """

    # FIX EVAL-6 — thresholds provenaient nulle part : report_data n'avait
    # jamais de clé "thresholds", donc reporting.include_thresholds=true
    # ne pouvait rien afficher même une fois branché côté Markdown. On les
    # lit depuis evaluation_config.yaml, seule source de vérité existante
    # pour ces valeurs (clinical_gate_status() les utilise déjà par
    # ailleurs pour calculer overall_status).
    eval_config = load_evaluation_config()
    thresholds = eval_config.get("thresholds", {})

    return {
        "model_name": model_name,
        "evaluation_timestamp": _utc_timestamp(),
        "overall_status": overall_status,
        "clinical_metrics": clinical_metrics,
        "safety": safety,
        "thresholds": thresholds,
        "metadata": metadata or {},
    }


# ============================================================
# EXPORT API
# ============================================================


def export_evaluation_reports(
    *,
    report_data: Dict[str, Any],
    output_dir: str | Path,
) -> Dict[str, str]:
    """
    Generate both JSON and Markdown reports.

    Returns:

    {
        "json_report": "...",
        "markdown_report": "..."
    }
    """

    json_path = save_json_report(
        report_data=report_data,
        output_dir=output_dir,
    )

    markdown_path = save_markdown_report(
        report_data=report_data,
        output_dir=output_dir,
    )

    return {
        "json_report": str(json_path),
        "markdown_report": str(markdown_path),
    }


# ============================================================
# CONVENIENCE API
# ============================================================


def generate_reports(
    *,
    model_name: str,
    clinical_metrics: Dict[str, Any],
    safety: Dict[str, Any],
    overall_status: str,
    output_dir: str | Path,
    metadata: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """
    High-level helper used by
    clinical_eval_runner.py.
    """

    report_data = create_evaluation_report(
        model_name=model_name,
        clinical_metrics=clinical_metrics,
        safety=safety,
        overall_status=overall_status,
        metadata=metadata,
    )

    files = export_evaluation_reports(
        report_data=report_data,
        output_dir=output_dir,
    )

    return {
        "report": report_data,
        "files": files,
    }
