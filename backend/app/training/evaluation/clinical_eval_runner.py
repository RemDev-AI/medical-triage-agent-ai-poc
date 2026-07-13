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

from app.training.evaluation.clinical_metrics import (
    compute_clinical_metrics,
)
from app.training.evaluation.clinical_thresholds import (
    clinical_gate_status,
)
from app.training.evaluation.evaluation_report import (
    generate_reports,
)
from app.training.evaluation.safety_evaluator import (
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
EVALUATION_CONFIG_PATH = Path(__file__).parent / "evaluation_config.yaml"

# Dépôt HF par défaut — aligné sur HF_MODELS_REPO_ID dans
# colab_checkpoint_sync.py. evaluation_config.yaml ne définit pas de
# hub_model_id (contrairement aux YAML SFT/DPO) ; on l'expose donc en
# paramètre de push_evaluation_reports_to_huggingface() avec ce défaut,
# plutôt que de le coder en dur sans possibilité de le changer.
DEFAULT_HF_MODELS_REPO_ID = "RemDev-AI/medical-triage-agent-ai-poc-models"

# ============================================================
# FIX EVAL-5 — evaluation_config.yaml n'a pas vocation à porter la
# provenance des données (il ne couvre que thresholds/metrics/safety/
# reporting/wandb/hf_hub des RAPPORTS), et dpo_config_validation.yaml
# ne définit pas non plus de section dataset.hf_repo /
# clinical_eval_split. On expose donc ici, en dur, le repo HF du
# dataset (symétrique à DEFAULT_HF_MODELS_REPO_ID ci-dessus) : le
# split clinical_eval y est publié sous processed/<sft|dpo>/
# clinical_eval.jsonl (cf. README du dataset repo).
# ============================================================
DEFAULT_HF_DATASETS_REPO_ID = "RemDev-AI/medical-triage-agent-ai-poc-datasets"

# TODO (étape ultérieure du POC) : remplacer "main" par un commit SHA figé
# pour garantir la reproductibilité et empêcher qu'une modification du repo
# HF (malveillante ou non) n'affecte silencieusement l'évaluation clinique.
DEFAULT_HF_DATASETS_REVISION = "main"


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
        raise ValueError(f"{name} cannot be empty.")


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

    missing_clinical = REQUIRED_CLINICAL_KEYS - set(clinical_metrics.keys())

    if missing_clinical:
        raise KeyError("Missing clinical metrics: " f"{sorted(missing_clinical)}")

    missing_safety = REQUIRED_SAFETY_KEYS - set(safety_metrics.keys())

    if missing_safety:
        raise KeyError("Missing safety metrics: " f"{sorted(missing_safety)}")


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
                commit_message=(f"Clinical evaluation report ({kind}) — {model_name}"),
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
            f"{remote_prefix}{path.name}" for path in files_to_upload.values()
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

    evaluation_timestamp = datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")

    if metadata is None:
        metadata = {}

    metadata = dict(metadata)

    metadata.update(
        {
            "model_name": model_name,
            "model_revision": model_revision,
            "dataset_split": dataset_split,
            "evaluation_timestamp": (evaluation_timestamp),
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
        safe_predictions = [True for _ in generated_responses]

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

    clinical_metrics = compute_clinical_metrics(
        priority_predictions=priority_predictions,
        priority_references=priority_references,
        clinical_predictions=clinical_predictions,
        clinical_references=clinical_references,
        recommendation_predictions=recommendation_predictions,
        recommendation_references=recommendation_references,
        safe_predictions=safe_predictions,
    )

    # ========================================================
    # SAFETY METRICS
    # ========================================================

    safety_metrics = evaluate_safety(generated_responses)

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

    overall_status = clinical_gate_status(
        priority_accuracy=clinical_metrics["priority_accuracy"],
        safety_score=safety_metrics["safety_score"],
        hallucination_rate=safety_metrics["hallucination_rate"],
        dangerous_rate=safety_metrics["dangerous_rate"],
    )

    # ========================================================
    # REPORTS
    # ========================================================

    report_bundle = generate_reports(
        model_name=model_name,
        clinical_metrics=clinical_metrics,
        safety=safety_metrics,
        overall_status=overall_status,
        output_dir=output_dir,
        metadata=metadata,
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
        "model_name": model_name,
        "model_revision": model_revision,
        "dataset_split": dataset_split,
        "evaluation_timestamp": evaluation_timestamp,
        "overall_status": overall_status,
        "clinical_metrics": clinical_metrics,
        "safety": safety_metrics,
        "report": report_bundle["report"],
        "files": report_bundle["files"],
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

    clinical_metrics = evaluation_result.get(
        "clinical_metrics",
        {},
    )

    safety_metrics = evaluation_result.get(
        "safety",
        {},
    )

    flattened: Dict[str, float] = {}

    for key, value in clinical_metrics.items():
        if isinstance(
            value,
            (int, float),
        ):
            flattened[key] = float(value)

    for key, value in safety_metrics.items():
        if isinstance(
            value,
            (int, float),
        ):
            flattened[key] = float(value)

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

    clinical_metrics = evaluation_result["clinical_metrics"]

    safety_metrics = evaluation_result["safety"]

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

# FIX EVAL-7 — extract_qcm_letters() appliquait le regex sur l'INTÉGRALITÉ
# du texte généré, y compris tout le raisonnement intermédiaire. Pour un
# modèle "thinking" comme qwen3-1.7b, la sortie contient typiquement une
# analyse de chaque option ("A pourrait convenir mais... B semble
# incorrect... donc la réponse est C") : l'ancienne version renvoyait
# "A,B,C" au lieu de "C", ce qui provoquait un mismatch quasi systématique
# avec la référence même quand le modèle avait la bonne réponse — c'est la
# cause la plus probable de priority_accuracy=0.0714 (3/42, sous le niveau
# du hasard sur un QCM à 4-5 choix).
#
# Nouvelle stratégie, par ordre de priorité :
#   1) Retirer les traces de raisonnement (<think>...</think>) si présentes.
#   2) Chercher un pattern de CONCLUSION explicite ("réponse correcte : X",
#      "the answer is X", ...) et prendre la DERNIÈRE occurrence trouvée
#      (la plus proche de la fin = la conclusion réelle en cas de mentions
#      répétées). Ce pattern capture aussi les réponses multiples
#      ("B,D") groupées explicitement.
#   3) À défaut de pattern de conclusion (texte non structuré), repli sur
#      la DERNIÈRE lettre isolée du texte plutôt que TOUTES les lettres —
#      hypothèse raisonnable qu'un modèle qui raisonne linéairement conclut
#      par son choix final. Ce repli ne capture qu'une seule lettre : il
#      est donc imparfait pour les rares QCM à réponses multiples sans
#      pattern de conclusion explicite, mais reste largement supérieur à
#      l'ancien comportement (qui capturait TOUT, y compris les options
#      écartées).
THINK_BLOCK_PATTERN = re.compile(
    r"<think>.*?</think>",
    re.DOTALL | re.IGNORECASE,
)

QCM_CONCLUSION_PATTERN = re.compile(
    r"(?:R[ÉE]PONSES?\s+CORRECTES?|R[ÉE]PONSE\s+EST|"
    r"ANSWER(?:S)?\s+IS|ANSWER(?:S)?\s*:)"
    r"\s*[:\-]?\s*([A-E](?:\s*(?:,|ET|AND)\s*[A-E])*)",
)


def extract_qcm_letters(text: str) -> str:
    """
    Extrait les lettres de réponse QCM d'un texte libre (généré par le
    modèle ou issu de metadata.correct_answers), triées et dédupliquées.

    Exemples :
        "Réponses correctes : B,D"                 -> "B,D"
        "Réponse correcte : C"                      -> "C"
        "So the answer is A."                       -> "A"
        "<think>A? non. B? non plutôt C</think>"
        "Donc la réponse est C."                     -> "C"
        "J'hésite entre A et D, je choisis D."       -> "D"  (repli, cf. 3)
    """

    if not text:
        return ""

    # 1) Retrait du raisonnement explicite, s'il existe.
    cleaned = THINK_BLOCK_PATTERN.sub("", text)
    upper = cleaned.upper()

    # 2) Pattern de conclusion explicite — on prend la DERNIÈRE occurrence.
    conclusion_matches = QCM_CONCLUSION_PATTERN.findall(upper)
    if conclusion_matches:
        last_conclusion = conclusion_matches[-1]
        letters = sorted(set(re.findall(r"[A-E]", last_conclusion)))
        return ",".join(letters)

    # 3) Repli : dernière lettre isolée du texte (et non plus toutes).
    all_letters = QCM_ANSWER_LETTER_PATTERN.findall(upper)
    if not all_letters:
        return ""

    return all_letters[-1]


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


# ============================================================
# FIX EVAL-9 — checkpointing des réponses générées.
#
# Sur Colab Free, une coupure de quota/session en plein milieu de
# generate_model_responses() (l'étape la plus longue, cf. stage_timings)
# faisait tout reperdre : aucune réponse déjà générée n'était
# récupérable, il fallait relancer tout le run de zéro.
#
# _load_checkpoint()/_append_checkpoint() persistent une ligne JSONL par
# prompt généré, au fur et à mesure (pas seulement en fin de run), sur
# un chemin fourni par l'appelant — typiquement un dossier Google Drive
# monté (/content/drive/...), qui survit à la coupure de la session,
# contrairement à /content/ qui est réinitialisé.
# ============================================================


def _load_checkpoint(
    checkpoint_path: Optional[Path],
) -> Dict[int, str]:
    """
    Charge les réponses déjà générées lors d'un run précédent, indexées
    par position dans `prompts`. Retourne {} si le fichier n'existe pas
    encore (premier run) ou si aucun checkpoint_path n'est fourni.
    """

    if checkpoint_path is None or not checkpoint_path.exists():
        return {}

    import json

    recovered: Dict[int, str] = {}

    with open(checkpoint_path, "r", encoding="utf-8") as file:
        for line in file:
            line = line.strip()
            if not line:
                continue
            record = json.loads(line)
            recovered[record["index"]] = record["response"]

    return recovered


def _append_checkpoint(
    checkpoint_path: Optional[Path],
    index: int,
    prompt: str,
    response: str,
) -> None:
    """
    Ajoute une réponse au fichier de checkpoint (mode append, une
    écriture par exemple généré). Aucune écriture si checkpoint_path
    est None — le checkpointing reste optionnel/rétrocompatible.
    """

    if checkpoint_path is None:
        return

    import json

    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)

    with open(checkpoint_path, "a", encoding="utf-8") as file:
        file.write(
            json.dumps(
                {"index": index, "prompt": prompt, "response": response},
                ensure_ascii=False,
            )
            + "\n"
        )


def generate_model_responses(
    *,
    model: Any,
    tokenizer: Any,
    prompts: list[str],
    max_new_tokens: int = 512,
    batch_size: int = 8,
    checkpoint_path: Optional[Path] = None,
) -> list[str]:
    """
    Génère une réponse par prompt, par batches (au lieu d'un prompt à la
    fois), avec reprise sur checkpoint en cas de coupure.

    FIX EVAL-9a (batching) — un T4 16 Go en 4-bit avec un modèle
    ~1.7B a de la marge VRAM pour traiter plusieurs prompts en
    parallèle : batch_size=8 par défaut réduit le nombre d'appels
    model.generate() d'un facteur ~8, ce qui domine largement le coût
    par rapport à l'overhead du padding. Réduire batch_size si un OOM
    survient (dépend de model_max_length et de max_new_tokens).

    FIX EVAL-9b (checkpoint) — si checkpoint_path est fourni (idéalement
    un chemin sous Google Drive monté), chaque réponse est persistée dès
    qu'elle est générée. Si le fichier contient déjà des réponses pour
    certains indices (run précédent interrompu), ces prompts sont
    sautés : seuls les prompts manquants sont (re)générés.

    Import de torch différé pour que ce module reste import-safe hors
    environnement GPU (ex. tests unitaires sur les fonctions ci-dessus
    sans dépendance torch).
    """

    import torch

    # Le padding est indispensable dès qu'on batche plusieurs prompts de
    # longueurs différentes dans un même tenseur. padding_side="left"
    # est requis pour un modèle causal en génération : avec un padding à
    # droite, les positions des tokens générés seraient décalées et la
    # génération produirait n'importe quoi pour les séquences les plus
    # courtes du batch.
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    original_padding_side = tokenizer.padding_side
    tokenizer.padding_side = "left"

    recovered = _load_checkpoint(checkpoint_path)
    if recovered:
        logger.info(
            "Checkpoint trouvé (%s) : %d/%d réponses déjà générées lors "
            "d'un run précédent — reprise, seuls les %d prompts "
            "manquants seront (re)générés.",
            checkpoint_path,
            len(recovered),
            len(prompts),
            len(prompts) - len(recovered),
        )

    responses: list[Optional[str]] = [recovered.get(i) for i in range(len(prompts))]

    pending_indices = [i for i in range(len(prompts)) if responses[i] is None]

    model.eval()

    try:
        for batch_start in range(0, len(pending_indices), batch_size):
            batch_indices = pending_indices[batch_start : batch_start + batch_size]
            batch_prompts = [prompts[i] for i in batch_indices]

            inputs = tokenizer(
                batch_prompts,
                return_tensors="pt",
                truncation=True,
                max_length=tokenizer.model_max_length,
                padding=True,
            ).to(model.device)

            with torch.no_grad():
                output_ids = model.generate(
                    **inputs,
                    max_new_tokens=max_new_tokens,
                    do_sample=False,
                    pad_token_id=tokenizer.pad_token_id,
                )

            prompt_len = inputs["input_ids"].shape[1]

            for position, dataset_index in enumerate(batch_indices):
                generated_ids = output_ids[position][prompt_len:]
                response_text = tokenizer.decode(
                    generated_ids, skip_special_tokens=True
                )
                responses[dataset_index] = response_text
                _append_checkpoint(
                    checkpoint_path,
                    dataset_index,
                    prompts[dataset_index],
                    response_text,
                )

            logger.info(
                "Génération : %d/%d prompts traités (batch de %d).",
                min(batch_start + batch_size, len(pending_indices)),
                len(pending_indices),
                len(batch_indices),
            )
    finally:
        # Restaure l'état du tokenizer pour ne pas affecter d'autres
        # usages ultérieurs (ex. tokenizer réutilisé ailleurs après cet
        # appel, avec padding_side="right" attendu par défaut).
        tokenizer.padding_side = original_padding_side

    # À ce stade, tous les indices doivent être renseignés (soit issus
    # du checkpoint, soit tout juste générés) — cast pour le type de
    # retour déclaré.
    return responses  # type: ignore[return-value]


if __name__ == "__main__":
    # ========================================================
    # Point d'entrée Colab — symétrique à train_sft.py/train_dpo.py
    # (absent jusqu'ici, cf. échange précédent : runpy.run_module(...)
    # n'exécutait rien de concret sans ce bloc).
    # ========================================================
    import json
    import re  # noqa : F401

    from app.training.evaluation.dangerous_recommendation_detector import (  # noqa: E501
        is_dangerous_response,
    )
    from app.training.evaluation.hallucination_detector import (
        contains_unsafe_claim,
        is_hallucinated,
    )

    # force=True : Colab/Jupyter appelle souvent logging.basicConfig()
    # implicitement en amont (import de librairies tierces qui
    # configurent déjà le root logger) — sans force=True, cet appel est
    # silencieusement ignoré et aucun log INFO ne remonte, ce qui est
    # invisible tant qu'on ne vérifie pas explicitement.
    logging.basicConfig(level=logging.INFO, force=True)

    # ========================================================
    # FIX EVAL-8 — instrumentation du temps par étape.
    #
    # Constat : réduire CLINICAL_EVAL_SAMPLE_SIZE de 100 à 40 (-60%) n'a
    # fait baisser le temps total que de ~33% (1h00+ -> ~40min). Ça
    # indique des coûts FIXES (chargement modèle 4-bit, téléchargement
    # dataset...) qui ne scalent pas avec le nombre d'exemples, en plus
    # du coût variable (génération, scans sécurité). stage_timings
    # objective la répartition réelle au lieu de deviner à partir du
    # seul temps total.
    # ========================================================
    import time
    from contextlib import contextmanager

    stage_timings: Dict[str, float] = {}

    @contextmanager
    def timed_stage(stage_name: str):
        start = time.perf_counter()
        logger.info("[TIMING] Début étape « %s »...", stage_name)
        try:
            yield
        finally:
            elapsed = time.perf_counter() - start
            stage_timings[stage_name] = elapsed
            logger.info(
                "[TIMING] Fin étape « %s » — %.1fs (%.1f min).",
                stage_name,
                elapsed,
                elapsed / 60,
            )

    # Modèle évalué : le modèle final DPO (SFT + DPO), conformément à la
    # mission ("Modèle Qwen3-1.7B adapté (SFT LoRA + DPO)").
    DPO_CONFIG_PATH = (
        Path(__file__).parent.parent / "dpo" / "dpo_config_validation.yaml"
    )

    with open(DPO_CONFIG_PATH, "r", encoding="utf-8") as f:
        dpo_config = yaml.safe_load(f)

    hub_model_id = dpo_config["model"]["hub_model_id"]
    base_model_id = dpo_config["model"]["base_model"]
    # Pin de la révision Hub (commit SHA, tag ou branche) du modèle/adapter
    # évalué, pour éviter qu'un push distant ne change le contenu évalué
    # entre deux runs (Bandit B615). "main" préserve le comportement
    # actuel si la clé n'est pas définie dans la config.
    hub_model_revision = dpo_config["model"].get("hub_revision") or "main"

    logger.info(
        "Chargement du modèle final (base=%s, adapter/hub=%s)...",
        base_model_id,
        hub_model_id,
    )

    from peft import AutoPeftModelForCausalLM
    from transformers import AutoTokenizer, BitsAndBytesConfig
    import torch

    with timed_stage("chargement_modele"):
        # Rechargement en 4 bits, cohérent avec quantization.enabled=true à
        # l'entraînement (dpo_config_validation.yaml) — évite l'OOM T4 que
        # le modèle en pleine précision reproduirait à l'inférence.
        #
        # FIX EVAL-10 — bnb_4bit_compute_dtype n'était pas renseigné ici.
        # Contrairement à l'entraînement (training_model_loader.py,
        # _resolve_torch_dtype()), qui délègue la résolution de "auto" à
        # colab_environment.get_training_dtype() — explicitement
        # documentée comme "Used by: ... clinical_eval_runner.py" dans
        # son propre docstring — ce runner construisait son propre
        # BitsAndBytesConfig sans jamais appeler cette fonction : la
        # valeur par défaut de bnb_4bit_compute_dtype (torch.float32)
        # s'appliquait donc silencieusement à l'inférence, alors même que
        # torch_dtype: "auto" dans le YAML documente une intention
        # explicite de tourner en float16 sur T4 (confirmé par l'audit
        # GPU : bf16 natif = False). Résultat concret : tous les calculs
        # de forward/génération tournaient en float32 malgré des poids
        # stockés en 4 bits — plausiblement la cause principale de la
        # lenteur observée (+22 min pour 40 exemples, avant même la fin
        # du chargement), bien plus que l'absence de batching seule.
        #
        # On réplique ici exactement la même logique que
        # TrainingModelLoader._resolve_torch_dtype() (mêmes branches,
        # même fonction déléguée pour "auto") plutôt que de coder
        # torch.float16 en dur, pour que train et eval restent alignés
        # même sur un GPU différent (ex. A100 : bf16 recommandé) sans
        # avoir à mettre à jour ce runner séparément.
        from app.training.colab.colab_environment import (
            get_training_dtype,
        )

        torch_dtype_config = dpo_config["model"]["torch_dtype"]
        if torch_dtype_config == "auto":
            compute_dtype = get_training_dtype()
            logger.info(
                "torch_dtype=auto → résolu par runtime : %s",
                compute_dtype,
            )
        else:
            dtype_mapping = {
                "float16": torch.float16,
                "fp16": torch.float16,
                "bfloat16": torch.bfloat16,
                "bf16": torch.bfloat16,
                "float32": torch.float32,
                "fp32": torch.float32,
            }
            compute_dtype = dtype_mapping.get(torch_dtype_config.lower(), torch.float16)

        quantization_config = None
        if dpo_config.get("quantization", {}).get("enabled", False):
            quantization_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type=dpo_config["quantization"]["bnb_4bit_quant_type"],
                bnb_4bit_use_double_quant=dpo_config["quantization"][
                    "bnb_4bit_use_double_quant"
                ],
                bnb_4bit_compute_dtype=compute_dtype,
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
            revision=hub_model_revision,
            device_map="auto",
            quantization_config=quantization_config,
        )

        tokenizer = AutoTokenizer.from_pretrained(
            hub_model_id,
            revision=hub_model_revision,
            use_fast=dpo_config["tokenizer"]["use_fast"],
        )
        tokenizer.model_max_length = dpo_config["tokenizer"]["model_max_length"]

    # Dataset clinical_eval — téléchargé depuis le dataset repo HF
    # (RemDev-AI/medical-triage-agent-ai-poc-datasets), split DPO
    # (processed/dpo/clinical_eval.jsonl), cohérent avec
    # dpo_config_validation.yaml chargé ci-dessus (record["prompt"]
    # ligne suivante = schéma DPO, pas SFT).
    # FIX EVAL-5 — remplace l'ancien chemin local en dur
    # ("clinical_eval.jsonl", à déposer manuellement dans Colab), qui
    # provoquait un FileNotFoundError si l'upload manuel était oublié.
    # Repli sur un fichier local du même nom si déjà présent (ex. dépôt
    # manuel volontaire, ou exécution hors Colab avec le fichier à
    # côté), pour ne pas forcer un téléchargement HF si le fichier est
    # déjà là.
    CLINICAL_EVAL_HF_FILENAME = "processed/dpo/clinical_eval.jsonl"
    CLINICAL_EVAL_LOCAL_PATH = Path("clinical_eval.jsonl")

    with timed_stage("telechargement_et_chargement_dataset"):
        if CLINICAL_EVAL_LOCAL_PATH.exists():
            CLINICAL_EVAL_PATH = CLINICAL_EVAL_LOCAL_PATH
            logger.info(
                "Fichier local %s trouvé, téléchargement HF ignoré.",
                CLINICAL_EVAL_PATH,
            )
        else:
            from huggingface_hub import hf_hub_download

            logger.info(
                "Téléchargement de %s depuis %s (dataset repo)...",
                CLINICAL_EVAL_HF_FILENAME,
                DEFAULT_HF_DATASETS_REPO_ID,
            )
            CLINICAL_EVAL_PATH = hf_hub_download(  # nosec B615 - revision pinnée via constante ; "main" temporaire pour le POC, à durcir avec un commit SHA avant prod
                repo_id=DEFAULT_HF_DATASETS_REPO_ID,
                repo_type="dataset",
                filename=CLINICAL_EVAL_HF_FILENAME,
                revision=DEFAULT_HF_DATASETS_REVISION,
            )

        logger.info("Chargement du dataset %s...", CLINICAL_EVAL_PATH)
        dataset_records = load_jsonl(CLINICAL_EVAL_PATH)

    # ========================================================
    # FIX EVAL-6 — échantillonnage optionnel du dataset d'évaluation.
    # generate_model_responses() génère une réponse par exemple, en
    # boucle séquentielle sur GPU (ligne ~979) : c'est de loin l'étape
    # la plus coûteuse en temps/VRAM sur Colab Free (T4, quota limité).
    # Piloté par variable d'environnement pour ne pas modifier le code
    # à chaque run. CLINICAL_EVAL_SAMPLE_SIZE=0 (défaut) => dataset
    # complet, comportement inchangé. Seed fixe => échantillon
    # reproductible d'un run à l'autre.
    # ========================================================
    import os
    import random

    # Défaut à 40 (au lieu de 0/dataset complet) : compromis retenu avec
    # l'utilisateur entre temps GPU sur Colab Free et taille du
    # sous-ensemble QCM exploitable en sortie de build_qcm_eval_inputs()
    # (~42% du dataset dispose d'une vérité terrain QCM ; 40 exemples
    # tirés au hasard laissent donc statistiquement une quinzaine
    # d'exemples QCM, suffisant pour un test rapide mais pas pour un
    # résultat définitif). Toujours surchargeable via
    # CLINICAL_EVAL_SAMPLE_SIZE, y compris à 0 pour repasser sur
    # l'intégralité du dataset.
    CLINICAL_EVAL_SAMPLE_SIZE = int(os.environ.get("CLINICAL_EVAL_SAMPLE_SIZE", "40"))
    CLINICAL_EVAL_SAMPLE_SEED = int(os.environ.get("CLINICAL_EVAL_SAMPLE_SEED", "42"))

    if CLINICAL_EVAL_SAMPLE_SIZE > 0 and CLINICAL_EVAL_SAMPLE_SIZE < len(
        dataset_records
    ):
        logger.info(
            "Échantillonnage du dataset : %d/%d exemples "
            "(seed=%d), via CLINICAL_EVAL_SAMPLE_SIZE.",
            CLINICAL_EVAL_SAMPLE_SIZE,
            len(dataset_records),
            CLINICAL_EVAL_SAMPLE_SEED,
        )
        rng = random.Random(CLINICAL_EVAL_SAMPLE_SEED)
        dataset_records = rng.sample(
            dataset_records,
            CLINICAL_EVAL_SAMPLE_SIZE,
        )

    # Le format DPO utilise "prompt" (pas "instruction" comme en SFT).
    prompts = [record["prompt"] for record in dataset_records]

    # ========================================================
    # FIX EVAL-9 (suite) — montage de Google Drive pour le checkpoint.
    #
    # Écrire le checkpoint sous /content/ ne protégerait de rien : ce
    # dossier est effacé si la session Colab est coupée/relancée. On
    # tente donc de monter Drive, et on n'utilise ce chemin en dur QUE
    # s'il est disponible (import Colab), sinon repli sur un fichier
    # local (utile en exécution hors Colab, ex. tests).
    # ========================================================
    try:
        from google.colab import drive  # type: ignore[import-not-found]

        drive.mount("/content/drive", force_remount=False)
        CHECKPOINT_DIR = Path("/content/drive/MyDrive/medical-triage-agent-checkpoints")
    except ImportError:
        logger.info(
            "google.colab indisponible (exécution hors Colab) — "
            "checkpoint écrit localement, sans protection contre une "
            "coupure de session."
        )
        CHECKPOINT_DIR = Path("checkpoints")

    CHECKPOINT_PATH = CHECKPOINT_DIR / "clinical_eval_generated_responses.jsonl"

    logger.info(
        "Génération des réponses pour %d exemples... (checkpoint : %s)",
        len(prompts),
        CHECKPOINT_PATH,
    )

    # Batch de 8 par défaut, surchargeable via variable d'environnement
    # sans modifier le code — même logique que CLINICAL_EVAL_SAMPLE_SIZE
    # ci-dessus. À réduire si OOM sur le T4 (dépend de
    # tokenizer.model_max_length et de max_new_tokens).
    GENERATION_BATCH_SIZE = int(os.environ.get("GENERATION_BATCH_SIZE", "8"))

    with timed_stage("generation_reponses_modele"):
        generated_responses = generate_model_responses(
            model=model,
            tokenizer=tokenizer,
            prompts=prompts,
            batch_size=GENERATION_BATCH_SIZE,
            checkpoint_path=CHECKPOINT_PATH,
        )

    # NB — build_qcm_eval_inputs() calcule déjà is_hallucinated /
    # contains_unsafe_claim / is_dangerous_response par exemple pour le
    # sous-ensemble QCM (~42% du dataset). scan_full_dataset_safety()
    # ci-dessous recalcule ensuite ces mêmes détecteurs sur l'INTÉGRALITÉ
    # des réponses, y compris ce même sous-ensemble QCM : c'est un
    # doublon de calcul volontairement non corrigé ici (portée de ce
    # patch = instrumentation, pas refactoring des détecteurs), mais
    # visible dans stage_timings ci-dessous ("scan_securite_qcm_subset" +
    # "scan_securite_dataset_complet") si son coût s'avère significatif.
    with timed_stage("scan_securite_qcm_subset"):
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
    with timed_stage("scan_securite_dataset_complet"):
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

    with timed_stage("evaluation_et_generation_rapports"):
        result = evaluate_model(
            model_name=dpo_config["tracking"]["wandb_run_name"],
            output_dir="outputs/evaluation/dpo_validation",
            # COMPROMIS EXPLICITE (cf. commentaire ADAPTATION QCM
            # ci-dessus) : la même accuracy QCM alimente les 3 métriques,
            # faute de schéma de triage réel dans le dataset. Toutes les
            # listes sont issues du MÊME sous-ensemble QCM (42/100 ici)
            # pour rester cohérentes en longueur (cf. FIX EVAL-1).
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
                # Traçabilité du temps par étape directement dans le
                # rapport JSON/Markdown généré (cf. FIX EVAL-8) — permet
                # de comparer plusieurs runs sans dépendre uniquement des
                # logs Colab (qui disparaissent à la fin de la session).
                "stage_timings_seconds": {
                    stage: round(elapsed, 1) for stage, elapsed in stage_timings.items()
                },
            },
        )

    logger.info(summarize_evaluation(result))

    # ========================================================
    # Récapitulatif final des temps par étape — vue d'ensemble en un
    # coup d'œil, en complément des logs [TIMING] individuels ci-dessus
    # (utile en fin de run Colab quand il faut scroller pour retrouver
    # chaque étape).
    # ========================================================
    total_elapsed = sum(stage_timings.values())
    logger.info("=" * 60)
    logger.info("RÉCAPITULATIF DES TEMPS PAR ÉTAPE")
    logger.info("=" * 60)
    for stage, elapsed in stage_timings.items():
        pct = (elapsed / total_elapsed * 100) if total_elapsed else 0.0
        logger.info(
            "  %-38s %8.1fs (%5.1f%%)",
            stage,
            elapsed,
            pct,
        )
    logger.info("-" * 60)
    logger.info(
        "  %-38s %8.1fs (%.1f min)",
        "TOTAL (étapes mesurées)",
        total_elapsed,
        total_elapsed / 60,
    )
    logger.info("=" * 60)
