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

import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from typing import Dict
from typing import Optional

# import json
import re  # noqa : F401

import yaml

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

logger = logging.getLogger(__name__)

# ============================================================
# FIX EVAL-4 — evaluation_config.yaml n'était chargé nulle part dans ce
# module malgré son en-tête ("Used by: clinical_eval_runner.py") : les
# clés hf_hub.enabled / attach_evaluation_reports / attach_markdown_summary
# n'avaient donc jamais d'effet, exactement comme push_to_hub dans les
# YAML SFT/DPO. On le charge ici pour piloter réellement la publication
# des rapports vers le Hub.
# ============================================================
EVALUATION_CONFIG_PATH = (
    Path(__file__).parent / "evaluation_config.yaml"
)

# Dépôt HF par défaut — aligné sur HF_MODELS_REPO_ID dans
# colab_checkpoint_sync.py. evaluation_config.yaml ne définit pas de
# hub_model_id (contrairement aux YAML SFT/DPO) ; on l'expose donc en
# paramètre de push_evaluation_reports_to_huggingface() avec ce défaut,
# plutôt que de le coder en dur sans possibilité de le changer.
DEFAULT_HF_MODELS_REPO_ID = (
    "RemDev-AI/medical-triage-agent-ai-poc-models"
)


def load_evaluation_config() -> Dict[str, Any]:
    with open(EVALUATION_CONFIG_PATH, "r", encoding="utf-8") as file:
        return yaml.safe_load(file)


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

    # FIX EVAL-1 — clinical_predictions et recommendation_predictions
    # n'étaient vérifiés que contre leurs propres references respectives,
    # jamais contre priority_predictions. Un désalignement de longueur
    # entre ces trois groupes de métriques passait donc inaperçu ici et
    # se répercutait silencieusement (zip tronqué, paires mal alignées)
    # dans compute_clinical_metrics() — risque critique dans un contexte
    # de triage clinique, où overall_status peut donner un feu vert basé
    # sur des paires prédiction/référence désalignées.
    _validate_same_length(
        "clinical_predictions",
        clinical_predictions,
        "priority_predictions",
        priority_predictions,
    )

    _validate_same_length(
        "recommendation_predictions",
        recommendation_predictions,
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
# FIX EVAL-4 (suite) — PUBLICATION DES RAPPORTS SUR HUGGING FACE
# ============================================================

def push_evaluation_reports_to_huggingface(
    *,
    report_bundle: Dict[str, Any],
    model_name: str,
    hub_model_id: str = DEFAULT_HF_MODELS_REPO_ID,
) -> None:
    """
    Publie les rapports JSON/Markdown générés par generate_reports() sur
    le dépôt Hugging Face, sous evaluation_reports/{model_name}/.

    Piloté par evaluation_config.yaml :
    - hf_hub.enabled : coupe-circuit global, aucun appel réseau si False.
    - hf_hub.attach_evaluation_reports : inclut le rapport JSON.
    - hf_hub.attach_markdown_summary : inclut le résumé Markdown.

    Contrairement au modèle final (SFT/DPO), les rapports d'évaluation
    sont de petits fichiers texte : ils sont conservés localement après
    upload (pas de nettoyage), l'objectif étant seulement de garantir
    qu'une copie existe aussi sur le Hub, pas de libérer de l'espace
    disque Colab.
    """

    eval_config = load_evaluation_config()
    hf_hub_config = eval_config.get("hf_hub", {})

    if not hf_hub_config.get("enabled", False):
        logger.info(
            "hf_hub.enabled=False dans evaluation_config.yaml — "
            "aucune publication des rapports d'évaluation."
        )
        return

    files_to_upload: Dict[str, Path] = {}

    report_files = report_bundle.get("files", {})

    if hf_hub_config.get("attach_evaluation_reports", False):
        json_path = report_files.get("json_report")
        if json_path:
            files_to_upload["json_report"] = Path(json_path)
        else:
            logger.warning(
                "hf_hub.attach_evaluation_reports=True mais aucun fichier "
                "JSON trouvé dans report_bundle['files']['json_report'] — "
                "rien à publier pour ce type."
            )

    if hf_hub_config.get("attach_markdown_summary", False):
        markdown_path = report_files.get("markdown_report")
        if markdown_path:
            files_to_upload["markdown_report"] = Path(markdown_path)
        else:
            logger.warning(
                "hf_hub.attach_markdown_summary=True mais aucun fichier "
                "Markdown trouvé dans report_bundle['files']['markdown_report'] "  # noqa: E501
                "— rien à publier pour ce type."
            )

    if not files_to_upload:
        logger.info(
            "Aucun rapport à publier sur Hugging Face (flags hf_hub.* "
            "tous désactivés ou fichiers absents)."
        )
        return

    from huggingface_hub import HfApi

    hf_api = HfApi()
    remote_prefix = f"evaluation_reports/{model_name}/"

    try:
        for kind, local_path in files_to_upload.items():
            if not local_path.exists():
                raise FileNotFoundError(
                    f"Fichier de rapport '{kind}' introuvable localement : "
                    f"{local_path}"
                )

            hf_api.upload_file(
                path_or_fileobj=str(local_path),
                path_in_repo=f"{remote_prefix}{local_path.name}",
                repo_id=hub_model_id,
                repo_type="model",
                commit_message=(
                    f"Clinical evaluation report ({kind}) — {model_name}"
                ),
            )

        # Vérification post-upload (aligné sur FIX HUB-3 / HUB-4) — on ne
        # fait pas confiance à la seule absence d'exception.
        remote_files = set(
            hf_api.list_repo_files(
                repo_id=hub_model_id,
                repo_type="model",
            )
        )

        expected_remote_names = {
            f"{remote_prefix}{path.name}"
            for path in files_to_upload.values()
        }

        missing = expected_remote_names - remote_files

        if missing:
            raise RuntimeError(
                f"Vérification post-upload échouée : rapport(s) absent(s) "
                f"du Hub après upload_file() : {sorted(missing)}"
            )

        logger.info(
            "Rapports d'évaluation publiés et vérifiés sur %s sous %s "
            "(%d fichier(s)).",
            hub_model_id,
            remote_prefix,
            len(files_to_upload),
        )

    except Exception:
        logger.exception(
            "Échec de la publication ou de la vérification des rapports "
            "d'évaluation sur Hugging Face — les fichiers locaux restent "
            "disponibles dans %s.",
            {str(p) for p in files_to_upload.values()},
        )
        raise


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
        datetime.now(timezone.utc)
        .isoformat()
        .replace("+00:00", "Z")
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
        # FIX EVAL-2 — ce défaut suppose que 100% des réponses sont sûres,
        # ce qui alimente silencieusement safety_accuracy dans
        # compute_clinical_metrics(). Un oubli d'intégration en amont
        # (appelant qui ne calcule/ne passe pas les vraies étiquettes de
        # sécurité) produirait un score parfait sans qu'aucun signal
        # n'attire l'attention dessus. On le rend au moins visible dans
        # les logs.
        logger.warning(
            "safe_predictions non fourni — défaut à [True] * %d exemples. "
            "safety_accuracy sera artificiellement à 100%% pour ce run si "
            "ce défaut n'est pas intentionnel.",
            len(generated_responses),
        )
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
    # FIX EVAL-4 (suite) — publication des rapports sur Hugging Face,
    # pilotée par evaluation_config.yaml (hf_hub.*), jusqu'ici jamais lue.
    # ========================================================
    push_evaluation_reports_to_huggingface(
        report_bundle=report_bundle,
        model_name=model_name,
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


# ============================================================
# ADAPTATION QCM — évaluation sur clinical_eval.jsonl réel
# ============================================================
#
# clinical_eval.jsonl (SFT et DPO) n'a AUCUNE colonne priority/clinical/
# recommendation (confirmé sur 3 fichiers : train.jsonl SFT, clinical_eval
# SFT et DPO). C'est un corpus de QCM/QA médicaux généraux
# (mediqa_mcqm, mediqa_mcqu, mediqa, ultramedical_preference, medquad).
#
# Décision actée avec l'utilisateur : adapter l'évaluation autour du seul
# signal fiable disponible — la correction QCM (metadata.correct_answers)
# — utilisée comme proxy d'accuracy clinique. La même valeur d'accuracy
# QCM alimente les 3 métriques priority/clinical/recommendation
# attendues par compute_clinical_metrics(), ce qui est un COMPROMIS
# EXPLICITE : ce n'est pas une vraie mesure de triage différenciée, mais
# cela permet de réutiliser tout le pipeline existant (seuils, rapports,
# push HF) sans le redessiner. Les contrôles de sécurité
# (hallucination/dangerous), eux, sont calculés sur TOUTES les réponses
# générées (QCM ou non), car ils ne nécessitent pas de vérité terrain.

QCM_ANSWER_LETTER_PATTERN = re.compile(r"\b([A-E])\b")


def extract_qcm_letters(text: str) -> str:
    """
    Extrait les lettres de réponse QCM d'un texte libre (généré par le
    modèle ou issu de metadata.correct_answers), triées et dédupliquées.

    Exemples :
        "Réponses correctes : B,D"  -> "B,D"
        "Réponse correcte : C"      -> "C"
        "So the answer is A."       -> "A"
    """

    if not text:
        return ""

    letters = sorted(set(QCM_ANSWER_LETTER_PATTERN.findall(text.upper())))

    return ",".join(letters)


def load_jsonl(path: str | Path) -> list[Dict[str, Any]]:
    records = []

    with open(path, "r", encoding="utf-8") as file:
        for line in file:
            line = line.strip()
            if line:
                records.append(json.loads(line))

    return records


def build_qcm_eval_inputs(
    *,
    dataset_records: list[Dict[str, Any]],
    generated_responses: list[str],
) -> Dict[str, Any]:
    """
    Construit les listes attendues par evaluate_model() à partir du
    dataset QCM réel + des réponses générées par le modèle.

    IMPORTANT — evaluate_model() exige que toutes les listes (priority/
    clinical/recommendation/generated_responses/safe_predictions) aient
    la MÊME longueur (cf. FIX EVAL-1). Comme seules ~42% des lignes ont
    une vérité terrain QCM exploitable (metadata.correct_answers), on
    filtre TOUT (prédictions QCM, réponses générées, safe_predictions)
    sur le même sous-ensemble cohérent — plutôt que de mélanger un
    sous-ensemble QCM avec la totalité des réponses générées.

    Les réponses hors QCM (58%) sont scannées séparément pour la
    sécurité via scan_full_dataset_safety() ci-dessous, car ce contrôle
    ne nécessite pas de vérité terrain.
    """

    qcm_predictions: list[str] = []
    qcm_references: list[str] = []
    qcm_generated_responses: list[str] = []
    safe_predictions: list[bool] = []

    for record, generated in zip(dataset_records, generated_responses):
        metadata = record.get("metadata", {})
        correct_answers = metadata.get("correct_answers")

        if not correct_answers:
            continue

        qcm_predictions.append(extract_qcm_letters(generated))
        qcm_references.append(extract_qcm_letters(correct_answers))
        qcm_generated_responses.append(generated)

        is_safe = not (
            is_hallucinated(generated)
            or contains_unsafe_claim(generated)
            or is_dangerous_response(generated)
        )
        safe_predictions.append(is_safe)

    return {
        "qcm_predictions": qcm_predictions,
        "qcm_references": qcm_references,
        "qcm_generated_responses": qcm_generated_responses,
        "safe_predictions": safe_predictions,
    }


def scan_full_dataset_safety(
    generated_responses: list[str],
) -> Dict[str, Any]:
    """
    Contrôle de sécurité sur l'INTÉGRALITÉ des réponses générées (QCM +
    texte libre), puisque hallucination/dangerous ne nécessitent pas de
    vérité terrain. Complémentaire au gate QCM de evaluate_model(), qui
    lui ne porte que sur le sous-ensemble QCM.
    """

    return evaluate_safety(generated_responses)


def generate_model_responses(
    *,
    model: Any,
    tokenizer: Any,
    prompts: list[str],
    max_new_tokens: int = 512,
) -> list[str]:
    """
    Génère une réponse par prompt. Import de torch différé pour que ce
    module reste important-safe hors environnement GPU (ex. tests
    unitaires sur les fonctions ci-dessus sans dépendance torch).
    """

    import torch

    responses = []

    model.eval()

    for prompt in prompts:
        inputs = tokenizer(
            prompt,
            return_tensors="pt",
            truncation=True,
            max_length=tokenizer.model_max_length,
        ).to(model.device)

        with torch.no_grad():
            output_ids = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=False,
            )

        generated_ids = output_ids[0][inputs["input_ids"].shape[1]:]

        responses.append(
            tokenizer.decode(generated_ids, skip_special_tokens=True)
        )

    return responses


if __name__ == "__main__":
    # ========================================================
    # Point d'entrée Colab — symétrique à train_sft.py/train_dpo.py
    # (absent jusqu'ici, cf. échange précédent : runpy.run_module(...)
    # n'exécutait rien de concret sans ce bloc).
    # ========================================================
    import json
    import re  # noqa : F401

    from backend.app.training.evaluation.dangerous_recommendation_detector import (  # noqa: E501
        is_dangerous_response,
    )
    from backend.app.training.evaluation.hallucination_detector import (
        contains_unsafe_claim,
        is_hallucinated,
    )

    logging.basicConfig(level=logging.INFO)

    # Modèle évalué : le modèle final DPO (SFT + DPO), conformément à la
    # mission ("Modèle Qwen3-1.7B adapté (SFT LoRA + DPO)").
    DPO_CONFIG_PATH = (
        Path(__file__).parent.parent / "dpo" / "dpo_config_validation.yaml"
    )

    with open(DPO_CONFIG_PATH, "r", encoding="utf-8") as f:
        dpo_config = yaml.safe_load(f)

    hub_model_id = dpo_config["model"]["hub_model_id"]
    base_model_id = dpo_config["model"]["base_model"]

    logger.info(
        "Chargement du modèle final (base=%s, adapter/hub=%s)...",
        base_model_id,
        hub_model_id,
    )

    from peft import AutoPeftModelForCausalLM
    from transformers import AutoTokenizer, BitsAndBytesConfig

    # Rechargement en 4 bits, cohérent avec quantization.enabled=true à
    # l'entraînement (dpo_config_validation.yaml) — évite l'OOM T4 que
    # le modèle en pleine précision reproduirait à l'inférence.
    quantization_config = None
    if dpo_config.get("quantization", {}).get("enabled", False):
        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type=dpo_config["quantization"][
                "bnb_4bit_quant_type"
            ],
            bnb_4bit_use_double_quant=dpo_config["quantization"][
                "bnb_4bit_use_double_quant"
            ],
        )

    # AutoPeftModelForCausalLM suppose que hub_model_id contient un
    # adapter LoRA (adapter_config.json + adapter_model.safetensors) —
    # cohérent avec lora.target_modules dans dpo_config_validation.yaml.
    # La racine de hub_model_id est réservée au modèle DPO de production
    # (cf. FIX HUB-COLLISION dans train_sft.py/train_dpo.py) — le modèle
    # SFT intermédiaire est publié séparément sous sft-final/, donc plus
    # de risque d'écrasement ici.
    model = AutoPeftModelForCausalLM.from_pretrained(
        hub_model_id,
        device_map="auto",
        quantization_config=quantization_config,
    )

    tokenizer = AutoTokenizer.from_pretrained(
        hub_model_id,
        use_fast=dpo_config["tokenizer"]["use_fast"],
    )
    tokenizer.model_max_length = dpo_config["tokenizer"]["model_max_length"]

    # Dataset clinical_eval — chemin local (déposé manuellement dans
    # Colab) ; adapter si le split est chargé depuis le Hub à la place
    # (dataset.hf_repo / dataset.clinical_eval_split dans le YAML).
    CLINICAL_EVAL_PATH = "clinical_eval.jsonl"

    logger.info("Chargement du dataset %s...", CLINICAL_EVAL_PATH)
    dataset_records = load_jsonl(CLINICAL_EVAL_PATH)

    # Le format DPO utilise "prompt" (pas "instruction" comme en SFT).
    prompts = [record["prompt"] for record in dataset_records]

    logger.info(
        "Génération des réponses pour %d exemples...", len(prompts)
    )
    generated_responses = generate_model_responses(
        model=model,
        tokenizer=tokenizer,
        prompts=prompts,
    )

    eval_inputs = build_qcm_eval_inputs(
        dataset_records=dataset_records,
        generated_responses=generated_responses,
    )

    qcm_predictions = eval_inputs["qcm_predictions"]
    qcm_references = eval_inputs["qcm_references"]
    qcm_generated_responses = eval_inputs["qcm_generated_responses"]
    safe_predictions = eval_inputs["safe_predictions"]

    logger.info(
        "%d/%d exemples avec vérité terrain QCM exploitable — "
        "evaluate_model() ne portera QUE sur ce sous-ensemble cohérent.",
        len(qcm_predictions),
        len(dataset_records),
    )

    # Scan de sécurité sur l'INTÉGRALITÉ du dataset (QCM + texte libre),
    # en complément du gate QCM ci-dessous qui ne couvre que 42% des
    # exemples. Ce scan n'entre pas dans overall_status (evaluate_model
    # ne le voit pas) — c'est un signal supplémentaire, loggé séparément.
    full_dataset_safety = scan_full_dataset_safety(generated_responses)
    logger.info(
        "Scan de sécurité sur l'intégralité du dataset (%d exemples) : "
        "hallucination_rate=%.4f, dangerous_rate=%.4f, "
        "unsafe_claim_rate=%.4f, safety_score=%.4f, thresholds_passed=%s",
        len(generated_responses),
        full_dataset_safety["hallucination_rate"],
        full_dataset_safety["dangerous_rate"],
        full_dataset_safety["unsafe_claim_rate"],
        full_dataset_safety["safety_score"],
        full_dataset_safety["thresholds_passed"],
    )

    result = evaluate_model(
        model_name=dpo_config["tracking"]["wandb_run_name"],
        output_dir="outputs/evaluation/dpo_validation",
        # COMPROMIS EXPLICITE (cf. commentaire ADAPTATION QCM ci-dessus) :
        # la même accuracy QCM alimente les 3 métriques, faute de schéma
        # de triage réel dans le dataset. Toutes les listes sont issues
        # du MÊME sous-ensemble QCM (42/100 ici) pour rester cohérentes
        # en longueur (cf. FIX EVAL-1).
        priority_predictions=qcm_predictions,
        priority_references=qcm_references,
        clinical_predictions=qcm_predictions,
        clinical_references=qcm_references,
        recommendation_predictions=qcm_predictions,
        recommendation_references=qcm_references,
        generated_responses=qcm_generated_responses,
        safe_predictions=safe_predictions,
        dataset_split="clinical_eval",
        model_revision=hub_model_id,
        metadata={
            "full_dataset_size": len(dataset_records),
            "qcm_subset_size": len(qcm_predictions),
            "full_dataset_safety_scan": full_dataset_safety,
        },
    )

    logger.info(summarize_evaluation(result))
