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
from datetime import datetime
from pathlib import Path
from typing import Any
from typing import Dict
from typing import Optional


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

    return datetime.utcnow().isoformat() + "Z"


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

    output_path = _ensure_directory(
        output_dir
    ) / filename

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

    for metric_name, metric_value in (
        clinical_metrics.items()
    ):
        lines.append(
            f"- **{metric_name}**: "
            f"{metric_value:.4f}"
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

    for metric_name, metric_value in (
        safety.items()
    ):
        if isinstance(
            metric_value,
            (float, int),
        ):
            lines.append(
                f"- **{metric_name}**: "
                f"{metric_value:.4f}"
            )
        else:
            lines.append(
                f"- **{metric_name}**: "
                f"{metric_value}"
            )

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
                else
                "Model does not satisfy clinical "
                "evaluation requirements."
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

    output_path = _ensure_directory(
        output_dir
    ) / filename

    markdown_content = (
        build_markdown_report(
            report_data
        )
    )

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
    metadata: Optional[
        Dict[str, Any]
    ] = None,
) -> Dict[str, Any]:
    """
    Build canonical report structure.
    """

    return {
        "model_name": model_name,
        "evaluation_timestamp":
            _utc_timestamp(),
        "overall_status":
            overall_status,
        "clinical_metrics":
            clinical_metrics,
        "safety":
            safety,
        "metadata":
            metadata or {},
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
        "json_report":
            str(json_path),
        "markdown_report":
            str(markdown_path),
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
    metadata: Optional[
        Dict[str, Any]
    ] = None,
) -> Dict[str, Any]:
    """
    High-level helper used by
    clinical_eval_runner.py.
    """

    report_data = (
        create_evaluation_report(
            model_name=model_name,
            clinical_metrics=clinical_metrics,
            safety=safety,
            overall_status=overall_status,
            metadata=metadata,
        )
    )

    files = export_evaluation_reports(
        report_data=report_data,
        output_dir=output_dir,
    )

    return {
        "report": report_data,
        "files": files,
    }
