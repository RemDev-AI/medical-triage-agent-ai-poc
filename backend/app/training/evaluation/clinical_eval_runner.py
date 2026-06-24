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
- HF Hub
- Weights & Biases
"""

from __future__ import annotations

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
    metadata: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """
    Main clinical evaluation entry point.

    Parameters
    ----------
    model_name:
        Evaluated model name.

    output_dir:
        Report destination directory.

    priority_predictions:
        Predicted triage priorities.

    priority_references:
        Ground-truth triage priorities.

    clinical_predictions:
        Predicted clinical labels.

    clinical_references:
        Ground-truth clinical labels.

    recommendation_predictions:
        Predicted recommendations.

    recommendation_references:
        Ground-truth recommendations.

    generated_responses:
        Model-generated responses used for
        hallucination and safety evaluation.

    safe_predictions:
        Optional safety labels.
        If omitted, inferred from safety metrics.

    metadata:
        Optional experiment metadata.

    Returns
    -------
    Dict[str, Any]
        Complete evaluation result.
    """

    # ========================================================
    # SAFETY LABELS
    # ========================================================

    if safe_predictions is None:
        safe_predictions = [
            True
            for _ in generated_responses
        ]

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
    # SAFETY EVALUATION
    # ========================================================

    safety_metrics = (
        evaluate_safety(
            generated_responses
        )
    )

    # ========================================================
    # GLOBAL CLINICAL GATE
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
    # REPORT GENERATION
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

    return {
        "model_name":
            model_name,
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
    Flatten metrics for W&B or MLflow logging.

    Example output:

    {
        "priority_accuracy": 0.91,
        "clinical_accuracy": 0.88,
        "recommendation_accuracy": 0.89,
        "safety_accuracy": 0.97,
        "hallucination_rate": 0.03,
        "dangerous_rate": 0.01,
        "safety_score": 0.98,
    }
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

    flattened = {}

    for key, value in (
        clinical_metrics.items()
    ):
        if isinstance(
            value,
            (float, int),
        ):
            flattened[key] = float(
                value
            )

    for key, value in (
        safety_metrics.items()
    ):
        if isinstance(
            value,
            (float, int),
        ):
            flattened[key] = float(
                value
            )

    return flattened


# ============================================================
# QUICK SUMMARY
# ============================================================

def summarize_evaluation(
    evaluation_result: Dict[str, Any],
) -> str:
    """
    Generate a short human-readable summary.
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
