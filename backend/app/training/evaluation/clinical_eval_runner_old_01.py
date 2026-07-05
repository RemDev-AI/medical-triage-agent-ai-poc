# medical-triage-agent-ai-poc/backend/app/training/evaluation/clinical_eval_runner.py

"""
Clinical evaluation entry point.

Responsibilities:
- evaluate_model()
- Compute clinical metrics
- Compute safety metrics
- Apply clinical thresholds
- Generate JSON report
- Generate Markdown report

Compatible with:
- SFT evaluation
- DPO evaluation
- Google Colab
- Hugging Face Hub
- Weights & Biases
- MLflow
"""

from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import Any
from typing import Dict
from typing import Optional

from backend.app.training.evaluation.clinical_metrics import (
    compute_clinical_metrics,
)
from backend.app.training.evaluation.clinical_thresholds import (
    clinical_gate_status,
)
from backend.app.training.evaluation.evaluation_report import (
    generate_reports,
)
from backend.app.training.evaluation.safety_evaluator import (
    evaluate_safety,
)


# ============================================================
# CONSTANTS
# ============================================================

REQUIRED_CLINICAL_KEYS = {
    "priority_accuracy",
    "clinical_accuracy",
    "recommendation_accuracy",
    "safety_accuracy",
}

REQUIRED_SAFETY_KEYS = {
    "hallucination_rate",
    "dangerous_rate",
    "safety_score",
}


# ============================================================
# VALIDATION HELPERS
# ============================================================

def _validate_non_empty(
    name: str,
    values: list[Any],
) -> None:
    """
    Validate non-empty list.
    """

    if not values:
        raise ValueError(
            f"{name} cannot be empty."
        )


def _validate_same_length(
    left_name: str,
    left_values: list[Any],
    right_name: str,
    right_values: list[Any],
) -> None:
    """
    Validate same-length arrays.
    """

    if len(left_values) != len(right_values):
        raise ValueError(
            f"Length mismatch: "
            f"{left_name}={len(left_values)} "
            f"!= "
            f"{right_name}={len(right_values)}"
        )


def _validate_inputs(
    *,
    priority_predictions: list[str],
    priority_references: list[str],
    clinical_predictions: list[str],
    clinical_references: list[str],
    recommendation_predictions: list[str],
    recommendation_references: list[str],
    generated_responses: list[str],
    safe_predictions: list[bool],
) -> None:
    """
    Validate evaluation inputs.
    """

    _validate_non_empty(
        "priority_predictions",
        priority_predictions,
    )

    _validate_non_empty(
        "priority_references",
        priority_references,
    )

    _validate_non_empty(
        "clinical_predictions",
        clinical_predictions,
    )

    _validate_non_empty(
        "clinical_references",
        clinical_references,
    )

    _validate_non_empty(
        "recommendation_predictions",
        recommendation_predictions,
    )

    _validate_non_empty(
        "recommendation_references",
        recommendation_references,
    )

    _validate_non_empty(
        "generated_responses",
        generated_responses,
    )

    _validate_same_length(
        "priority_predictions",
        priority_predictions,
        "priority_references",
        priority_references,
    )

    _validate_same_length(
        "clinical_predictions",
        clinical_predictions,
        "clinical_references",
        clinical_references,
    )

    _validate_same_length(
        "recommendation_predictions",
        recommendation_predictions,
        "recommendation_references",
        recommendation_references,
    )

    _validate_same_length(
        "generated_responses",
        generated_responses,
        "priority_predictions",
        priority_predictions,
    )

    _validate_same_length(
        "safe_predictions",
        safe_predictions,
        "priority_predictions",
        priority_predictions,
    )


def _validate_metric_keys(
    clinical_metrics: Dict[str, Any],
    safety_metrics: Dict[str, Any],
) -> None:
    """
    Validate required metric keys.
    """

    missing_clinical = (
        REQUIRED_CLINICAL_KEYS
        - set(clinical_metrics.keys())
    )

    if missing_clinical:
        raise KeyError(
            "Missing clinical metrics: "
            f"{sorted(missing_clinical)}"
        )

    missing_safety = (
        REQUIRED_SAFETY_KEYS
        - set(safety_metrics.keys())
    )

    if missing_safety:
        raise KeyError(
            "Missing safety metrics: "
            f"{sorted(missing_safety)}"
        )


# ============================================================
# CLINICAL EVALUATION
# ============================================================

def evaluate_model(
    *,
    model_name: str,
    output_dir: str | Path,
    priority_predictions: list[str],
    priority_references: list[str],
    clinical_predictions: list[str],
    clinical_references: list[str],
    recommendation_predictions: list[str],
    recommendation_references: list[str],
    generated_responses: list[str],
    safe_predictions: Optional[list[bool]] = None,
    dataset_split: str = "clinical_eval",
    model_revision: Optional[str] = None,
    metadata: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """
    Main clinical evaluation entry point.
    """

    evaluation_timestamp = (
        datetime.utcnow().isoformat()
        + "Z"
    )

    if metadata is None:
        metadata = {}

    metadata = dict(metadata)

    metadata.update(
        {
            "model_name": model_name,
            "model_revision": model_revision,
            "dataset_split": dataset_split,
            "evaluation_timestamp": (
                evaluation_timestamp
            ),
        }
    )

    # ========================================================
    # SAFETY LABELS
    # ========================================================

    if safe_predictions is None:
        safe_predictions = [
            True
            for _ in generated_responses
        ]

    # ========================================================
    # INPUT VALIDATION
    # ========================================================

    _validate_inputs(
        priority_predictions=priority_predictions,
        priority_references=priority_references,
        clinical_predictions=clinical_predictions,
        clinical_references=clinical_references,
        recommendation_predictions=recommendation_predictions,
        recommendation_references=recommendation_references,
        generated_responses=generated_responses,
        safe_predictions=safe_predictions,
    )

    # ========================================================
    # CLINICAL METRICS
    # ========================================================

    clinical_metrics = (
        compute_clinical_metrics(
            priority_predictions=priority_predictions,
            priority_references=priority_references,
            clinical_predictions=clinical_predictions,
            clinical_references=clinical_references,
            recommendation_predictions=recommendation_predictions,
            recommendation_references=recommendation_references,
            safe_predictions=safe_predictions,
        )
    )

    # ========================================================
    # SAFETY METRICS
    # ========================================================

    safety_metrics = (
        evaluate_safety(
            generated_responses
        )
    )

    # ========================================================
    # VALIDATE METRICS
    # ========================================================

    _validate_metric_keys(
        clinical_metrics,
        safety_metrics,
    )

    # ========================================================
    # CLINICAL GATE
    # ========================================================

    overall_status = (
        clinical_gate_status(
            priority_accuracy=clinical_metrics[
                    "priority_accuracy"
                ],
            safety_score=safety_metrics[
                    "safety_score"
                ],
            hallucination_rate=safety_metrics[
                    "hallucination_rate"
                ],
            dangerous_rate=safety_metrics[
                    "dangerous_rate"
                ],
        )
    )

    # ========================================================
    # REPORTS
    # ========================================================

    report_bundle = (
        generate_reports(
            model_name=model_name,
            clinical_metrics=clinical_metrics,
            safety=safety_metrics,
            overall_status=overall_status,
            output_dir=output_dir,
            metadata=metadata,
        )
    )

    # ========================================================
    # RESULT
    # ========================================================

    return {
        "model_name":
            model_name,
        "model_revision":
            model_revision,
        "dataset_split":
            dataset_split,
        "evaluation_timestamp":
            evaluation_timestamp,
        "overall_status":
            overall_status,
        "clinical_metrics":
            clinical_metrics,
        "safety":
            safety_metrics,
        "report":
            report_bundle["report"],
        "files":
            report_bundle["files"],
    }


# ============================================================
# W&B / MLFLOW FRIENDLY FLATTENING
# ============================================================

def flatten_metrics(
    evaluation_result: Dict[str, Any],
) -> Dict[str, float]:
    """
    Flatten metrics for W&B and MLflow.
    """

    clinical_metrics = (
        evaluation_result.get(
            "clinical_metrics",
            {},
        )
    )

    safety_metrics = (
        evaluation_result.get(
            "safety",
            {},
        )
    )

    flattened: Dict[str, float] = {}

    for key, value in (
        clinical_metrics.items()
    ):
        if isinstance(
            value,
            (int, float),
        ):
            flattened[key] = float(
                value
            )

    for key, value in (
        safety_metrics.items()
    ):
        if isinstance(
            value,
            (int, float),
        ):
            flattened[key] = float(
                value
            )

    return flattened


# ============================================================
# HUMAN SUMMARY
# ============================================================

def summarize_evaluation(
    evaluation_result: Dict[str, Any],
) -> str:
    """
    Human-readable summary.
    """

    clinical_metrics = (
        evaluation_result[
            "clinical_metrics"
        ]
    )

    safety_metrics = (
        evaluation_result[
            "safety"
        ]
    )

    return (
        f"Status="
        f"{evaluation_result['overall_status']} | "
        f"Priority="
        f"{clinical_metrics['priority_accuracy']:.4f} | "
        f"Clinical="
        f"{clinical_metrics['clinical_accuracy']:.4f} | "
        f"Recommendation="
        f"{clinical_metrics['recommendation_accuracy']:.4f} | "
        f"Safety="
        f"{safety_metrics['safety_score']:.4f}"
    )
