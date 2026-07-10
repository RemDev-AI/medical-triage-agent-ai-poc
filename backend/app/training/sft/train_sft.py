# medical-triage-agent-ai-poc/backend/app/training/sft/train_sft.py

# Version : validation LoRA — bugs #1 #2 #3 corrigés
#
# Correctif additionnel (audit OOM DPO, analyse train_sft.py) :
#   - GC-1 : gradient_checkpointing_kwargs ajouté à TrainingArguments.
#            Sans ce kwarg, le Trainer HF réactive le gradient checkpointing
#            au démarrage de trainer.train() (car training_config
#            ["gradient_checkpointing"]=True), en écrasant potentiellement
#            le use_reentrant=False déjà posé par
#            TrainingModelLoader.apply_gradient_checkpointing() (bug #4,
#            requis pour Qwen3). Le rendre explicite ici garantit la
#            cohérence même si le Trainer réactive le GC.
#   - TOK-1/TOK-2 : modèle chargé AVANT le tokenizer (nécessaire pour la
#            branche Unsloth, qui charge les deux ensemble), et
#            synchronisation explicite model/tokenizer (pad/eos/bos ids)
#            pour éliminer le warning PAD/BOS/EOS à la source.
#   - DTYPE-1 : ajout de ForceMasterWeightsFp32Callback (parité avec le
#            correctif appliqué à train_dpo.py le 2026-07-05). Ce script
#            utilise le même TrainingModelLoader (branche Unsloth possible)
#            sur le même T4 avec la même politique fp16/GradScaler ; sans
#            ce filet de sécurité à on_train_begin, il est exposé au même
#            risque de crash "Attempting to unscale FP16 gradients" que
#            celui diagnostiqué et corrigé côté DPO.

from __future__ import annotations

import json
import logging
import math
from pathlib import Path
from typing import Dict, List, Tuple

import torch
import yaml
from datasets import Dataset, load_dataset
from transformers import (
    DataCollatorForSeq2Seq,
    EarlyStoppingCallback,
    Trainer,
    TrainerCallback,
    TrainingArguments,
)

from backend.app.training.colab.colab_environment import (
    apply_precision_arguments,
    resolve_quantization_settings,
)
from backend.app.training.shared.training_model_loader import (
    TrainingModelLoader,
)
from backend.app.training.shared.training_tokenizer_loader import (
    TrainingTokenizerLoader,
)
from backend.app.training.shared.training_utils import (
    TrainingUtils,
)
from backend.app.training.colab.colab_checkpoint_sync import (
    create_default_checkpoint_sync,
)

logger = logging.getLogger(__name__)

CONFIG_PATH = Path(__file__).parent / "sft_config_validation.yaml"


def load_config() -> Dict:
    with open(CONFIG_PATH, "r", encoding="utf-8") as file:
        return yaml.safe_load(file)


CONFIG = load_config()
# FIX QUANT-2 — complète config["quantization"] selon le GPU détecté si
# absent du YAML ; conserve tel quel un choix explicite déjà présent
# (avec warning informatif en cas de désaccord). cf. colab_environment.py.
CONFIG = resolve_quantization_settings(CONFIG)


# ---------------------------------------------------------------------------
# FIX BUG #5 — NaNGuardCallback
# Arrêt immédiat si eval_loss = NaN pour éviter un run silencieusement cassé
# ---------------------------------------------------------------------------
class NaNGuardCallback(TrainerCallback):
    def on_evaluate(self, args, state, control, metrics, **kwargs):
        loss = metrics.get("eval_loss", 0.0)
        if math.isnan(loss) or math.isinf(loss):
            raise ValueError(
                f"eval_loss={loss} détecté à step {state.global_step}. "
                "Arrêt du run. Vérifier le masquage des labels (bug #1)."
            )


# ---------------------------------------------------------------------------
# FIX DTYPE-1 — ForceMasterWeightsFp32Callback (parité avec train_dpo.py)
#
# Contexte (cf. audit DPO du 2026-07-05) : sur T4, la politique de précision
# impose fp16=True/bf16=False + GradScaler fp16. Or TrainingModelLoader
# (branche Unsloth) peut réintroduire du bfloat16 sur les poids maîtres LoRA
# après get_peft_model(), car Unsloth applique en interne sa propre
# détection GPU (is_bfloat16_supported()), indépendante du dtype demandé.
# Cette réintroduction se produit AVANT trainer.train() et n'est donc pas
# spécifique à DPOTrainer : ce Trainer standard est exposé au même risque de
# crash "Attempting to unscale FP16 gradients" si des paramètres
# entraînables se retrouvent en bf16 (ou fp16 mal placé) au moment où le
# GradScaler fp16 intervient. Ce callback est le filet de sécurité final,
# exécuté juste avant le début de l'entraînement, qui force tout paramètre
# entraînable en float32, seul dtype de stockage compatible avec un
# GradScaler fp16.
# ---------------------------------------------------------------------------
class ForceMasterWeightsFp32Callback(TrainerCallback):
    def on_train_begin(self, args, state, control, model=None, **kwargs):
        target_model = model if model is not None else kwargs.get("model")
        if target_model is None:
            logger.warning(
                "ForceMasterWeightsFp32Callback : modèle introuvable dans "
                "on_train_begin, aucun recast effectué."
            )
            return

        recast_count = 0
        for name, param in target_model.named_parameters():
            if param.requires_grad and param.dtype in (
                torch.bfloat16,
                torch.float16,
            ):
                param.data = param.data.to(torch.float32)
                recast_count += 1

        if recast_count:
            logger.warning(
                "ForceMasterWeightsFp32Callback : %d paramètre(s) "
                "entraînable(s) recastés vers float32 avant le début de "
                "l'entraînement (bf16/fp16 détectés).",
                recast_count,
            )


def load_jsonl_dataset(path: str) -> Dataset:
    records: List[Dict] = []
    with open(path, "r", encoding="utf-8") as file:
        for line in file:
            records.append(json.loads(line))
    return Dataset.from_list(records)


def load_hf_dataset(
    dataset_repo: str,
    dataset_config: str,
    split: str,
    revision: str = "main",
) -> Dataset:
    # TODO (étape ultérieure du POC) : remplacer "main" (valeur par défaut ci-dessus)
    # par un commit SHA figé pour garantir la reproductibilité et empêcher qu'une
    # modification du repo HF n'affecte silencieusement l'entraînement.
    return load_dataset(  # nosec B615 - revision pinnée via paramètre ; "main" temporaire pour le POC, à durcir avant prod
        path=dataset_repo,
        name=dataset_config,
        split=split,
        revision=revision,
    )


def load_dataset_source(split: str) -> Dataset:
    dataset_config = CONFIG["dataset"]

    hf_repo = dataset_config.get("hf_repo")
    hf_config = dataset_config.get("hf_config", "sft")
    hf_revision = dataset_config.get("hf_revision", "main")

    if hf_repo:
        dataset = load_hf_dataset(
            dataset_repo=hf_repo,
            dataset_config=hf_config,
            split=split,
            revision=hf_revision,
        )
    else:
        path_mapping = {
            "train": "train_path",
            "validation": "validation_path",
            "test": "test_path",
            "clinical_eval": "clinical_eval_path",
        }
        dataset = load_jsonl_dataset(dataset_config[path_mapping[split]])

    # Sous-ensemble pour run de validation rapide
    max_key = "max_train_samples" if split == "train" else "max_val_samples"
    max_samples = dataset_config.get(max_key)
    if max_samples:
        dataset = dataset.select(range(min(max_samples, len(dataset))))
        logger.info("Split '%s' limité à %d exemples.", split, len(dataset))

    return dataset


def build_prompt(example: Dict) -> Dict:
    prompt = (
        "<|system|>\n"
        "You are a medical triage assistant.\n"
        "Provide clinically safe recommendations.\n"
        "<|user|>\n"
        f"{example['instruction']}\n"
        "<|assistant|>\n"
        f"{example['response']}"
    )
    return {"text": prompt}


# ---------------------------------------------------------------------------
# FIX BUG #1 — tokenize_function
# Vérifie qu'au moins un token est actif après masquage.
# Retourne None pour les exemples entièrement masqués (filtrés ensuite).
# ---------------------------------------------------------------------------
def tokenize_function(examples, tokenizer, max_length: int):
    outputs = tokenizer(
        examples["text"],
        truncation=True,
        padding=False,
        max_length=max_length,
    )

    labels = []
    skipped = 0

    for input_ids, text in zip(outputs["input_ids"], examples["text"]):
        assistant_marker = "<|assistant|>\n"
        parts = text.split(assistant_marker)

        if len(parts) < 2:
            logger.warning("Marqueur <|assistant|> absent — exemple masqué.")
            labels.append([-100] * len(input_ids))
            skipped += 1
            continue

        prefix = parts[0] + assistant_marker
        prefix_ids = tokenizer(
            prefix,
            truncation=True,
            max_length=max_length,
            add_special_tokens=False,
        )["input_ids"]

        prefix_len = min(len(prefix_ids), len(input_ids))
        label = [-100] * prefix_len + input_ids[prefix_len:]

        # FIX : vérification des tokens actifs
        active_tokens = sum(1 for t in label if t != -100)
        if active_tokens == 0:
            logger.warning(
                "Aucun token actif après masquage (prefix_len=%d, "
                "input_ids_len=%d) — exemple écarté.",
                prefix_len,
                len(input_ids),
            )
            # On conserve la séquence pour le filtre suivant
            labels.append([-100] * len(input_ids))
            skipped += 1
            continue

        labels.append(label)

    if skipped:
        logger.warning(
            "%d exemple(s) entièrement masqués dans ce batch — " "ils seront filtrés.",
            skipped,
        )

    outputs["labels"] = labels
    return outputs


def prepare_datasets(tokenizer) -> Tuple[Dataset, Dataset]:
    dataset_config = CONFIG["dataset"]

    train_dataset = load_dataset_source("train")
    validation_dataset = load_dataset_source("validation")

    train_dataset = train_dataset.map(
        build_prompt,
        remove_columns=train_dataset.column_names,
    )
    validation_dataset = validation_dataset.map(
        build_prompt,
        remove_columns=validation_dataset.column_names,
    )

    train_dataset = train_dataset.map(
        lambda x: tokenize_function(
            x, tokenizer, dataset_config["max_sequence_length"]
        ),
        batched=True,
    )
    validation_dataset = validation_dataset.map(
        lambda x: tokenize_function(
            x, tokenizer, dataset_config["max_sequence_length"]
        ),
        batched=True,
    )

    # FIX BUG #1 — filtre les exemples sans token actif
    before_train = len(train_dataset)
    before_val = len(validation_dataset)

    train_dataset = train_dataset.filter(lambda x: any(t != -100 for t in x["labels"]))
    validation_dataset = validation_dataset.filter(
        lambda x: any(t != -100 for t in x["labels"])
    )

    logger.info(
        "Train : %d → %d exemples après filtre (-%d masqués).",
        before_train,
        len(train_dataset),
        before_train - len(train_dataset),
    )
    logger.info(
        "Validation : %d → %d exemples après filtre (-%d masqués).",
        before_val,
        len(validation_dataset),
        before_val - len(validation_dataset),  # noqa : E501
    )

    if len(validation_dataset) == 0:
        raise ValueError(
            "Validation dataset vide après filtre. "
            "Tous les labels étaient masqués — vérifier build_prompt() "
            "et le marqueur <|assistant|>."
        )

    return train_dataset, validation_dataset


def build_training_arguments() -> TrainingArguments:
    training_config = CONFIG["training"]

    training_args = {
        "output_dir": training_config["output_dir"],
        "num_train_epochs": training_config["num_train_epochs"],
        "per_device_train_batch_size": training_config[
            "per_device_train_batch_size"
        ],  # noqa : E501
        "per_device_eval_batch_size": training_config[
            "per_device_eval_batch_size"
        ],  # noqa : E501
        "gradient_accumulation_steps": training_config[
            "gradient_accumulation_steps"
        ],  # noqa : E501
        "learning_rate": float(training_config["learning_rate"]),
        "weight_decay": float(training_config["weight_decay"]),
        "warmup_ratio": float(training_config["warmup_ratio"]),
        "logging_steps": training_config["logging_steps"],
        "eval_steps": training_config["eval_steps"],
        "save_steps": training_config["save_steps"],
        "save_total_limit": training_config["save_total_limit"],
        "eval_strategy": training_config["evaluation_strategy"],
        "save_strategy": training_config["save_strategy"],
        "load_best_model_at_end": training_config["load_best_model_at_end"],
        "metric_for_best_model": training_config["metric_for_best_model"],
        "greater_is_better": training_config["greater_is_better"],
        "gradient_checkpointing": training_config["gradient_checkpointing"],
        # FIX GC-1 — le Trainer HF réactive lui-même le gradient
        # checkpointing au démarrage de trainer.train() (puisque
        # gradient_checkpointing=True ci-dessus), en écrasant
        # potentiellement le use_reentrant=False déjà posé par
        # TrainingModelLoader.apply_gradient_checkpointing() (bug #4,
        # requis pour Qwen3). On le rend explicite ici pour rester
        # cohérent quelle que soit la source de l'activation.
        "gradient_checkpointing_kwargs": {"use_reentrant": False},
        "lr_scheduler_type": training_config["lr_scheduler_type"],
        "max_grad_norm": float(training_config["max_grad_norm"]),
        "report_to": training_config.get("report_to", ["wandb"]),
        "dataloader_num_workers": training_config.get(
            "dataloader_num_workers", 0
        ),  # noqa : E501
        "remove_unused_columns": True,
    }

    # FIX BUG #3 — source unique de précision : le runtime détecte le GPU
    training_args = apply_precision_arguments(training_args)

    return TrainingArguments(**training_args)


def build_trainer(model, tokenizer, train_dataset, validation_dataset):
    training_args = build_training_arguments()

    data_collator = DataCollatorForSeq2Seq(
        tokenizer=tokenizer,
        model=model,
        padding=True,
        pad_to_multiple_of=8,
        label_pad_token_id=-100,
    )

    early_stopping = EarlyStoppingCallback(
        early_stopping_patience=CONFIG["training"]["early_stopping_patience"]
    )

    # FIX BUG #5 — NaNGuardCallback en premier pour stopper avant EarlyStopping
    nan_guard = NaNGuardCallback()

    # FIX DTYPE-1 — doit s'exécuter à on_train_begin, avant tout forward/
    # backward, pour garantir que le GradScaler fp16 ne voit jamais de
    # poids maîtres entraînables en bf16.
    dtype_guard = ForceMasterWeightsFp32Callback()

    return Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=validation_dataset,
        processing_class=tokenizer,
        data_collator=data_collator,
        callbacks=[dtype_guard, nan_guard, early_stopping],
    )


def publish_training_artifacts() -> None:
    output_dir = Path(CONFIG["training"]["output_dir"])
    metadata = TrainingUtils.build_training_metadata(config=CONFIG)
    TrainingUtils.save_training_metadata(
        metadata=metadata,
        output_directory=output_dir,
    )
    logger.info("Training metadata saved.")


def train(publish_to_hf: bool = False):  # False par défaut en validation
    TrainingUtils.setup_logging(CONFIG["system"]["logging_level"])
    logger.info("Starting SFT validation run.")

    TrainingUtils.set_seed(CONFIG["system"]["seed"])

    wandb_run = TrainingUtils.initialize_wandb(config=CONFIG)

    # FIX TOK-1 — le modèle est chargé EN PREMIER : en mode
    # runtime.engine="unsloth", TrainingModelLoader charge modèle ET
    # tokenizer ensemble (FastLanguageModel.from_pretrained). Le tokenizer
    # pré-chargé est ensuite réutilisé par TrainingTokenizerLoader au lieu
    # d'en recharger un indépendant (source d'incohérence sinon).
    model_loader = TrainingModelLoader(CONFIG)
    model = model_loader.prepare_for_training()

    tokenizer = TrainingTokenizerLoader.build(
        config=CONFIG,
        preloaded_tokenizer=getattr(model_loader, "unsloth_tokenizer", None),
    )

    # FIX TOK-2 — synchronise explicitement model.config /
    # model.generation_config avec le tokenizer (élimine le warning
    # "tokenizer has new PAD/BOS/EOS tokens" à la source).
    TrainingTokenizerLoader.sync_with_model(model=model, tokenizer=tokenizer)

    train_dataset, validation_dataset = prepare_datasets(tokenizer=tokenizer)

    logger.info(
        "Validation run — train=%d exemples, val=%d exemples.",
        len(train_dataset),
        len(validation_dataset),
    )

    trainer = build_trainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=train_dataset,
        validation_dataset=validation_dataset,
    )

    checkpoint_sync = create_default_checkpoint_sync(
        output_dir=CONFIG["training"]["output_dir"],
        training_type="sft",
    )

    resume_checkpoint = CONFIG["training"].get("resume_from_checkpoint")

    if resume_checkpoint is None:
        resume_checkpoint = checkpoint_sync.restore_latest_checkpoint_from_huggingface()

    trainer.train(
        # resume_from_checkpoint=CONFIG["training"].get("resume_from_checkpoint")
        resume_from_checkpoint=resume_checkpoint
    )

    trainer.save_model()
    tokenizer.save_pretrained(CONFIG["training"]["output_dir"])
    publish_training_artifacts()

    # FIX HUB-1 — publish_to_hf était déclaré mais jamais utilisé : aucun
    # push vers le Hub n'était possible quelle que soit sa valeur.
    #
    # FIX HUB-2 — checkpoint_sync.sync_*_to_huggingface() n'uploade QUE les
    # sous-dossiers "checkpoint-<step>/" créés par le Trainer lorsque
    # save_strategy="steps" declenche une sauvegarde (step % save_steps == 0).
    # trainer.save_model() ci-dessus sauvegarde en revanche directement dans
    # output_dir/ (le modèle final), qui n'est PAS un dossier "checkpoint-*".
    # Les deux mécanismes sont donc complémentaires, pas interchangeables :
    #   - checkpoint_sync -> synchronise les checkpoints INTERMEDIAIRES
    #     (utile pour resume_from_checkpoint en cas de coupure Colab)
    #   - upload_folder direct -> publie le modèle FINAL sur hub_model_id
    if publish_to_hf:
        # 1) Checkpoints intermédiaires éventuels (utile si save_strategy
        #    a effectivement déclenché une sauvegarde pendant le run).
        #    sync_all_checkpoints_to_huggingface() nettoie déjà chaque
        #    checkpoint-N/ localement après upload réussi (cf.
        #    cleanup_after_upload=True par défaut dans ColabCheckpointSync).
        if checkpoint_sync.has_checkpoint():
            logger.info(
                "Synchronisation des checkpoints intermédiaires vers Hugging Face..."
            )  # noqa: E501
            # FIX HUB-5 — la valeur de retour était auparavant ignorée :
            # un échec de push (ou d'un des deux, en cas de plusieurs
            # checkpoints) passait totalement inaperçu.
            checkpoints_synced = (
                checkpoint_sync.sync_all_checkpoints_to_huggingface()
            )  # noqa: E501
            if not checkpoints_synced:
                raise RuntimeError(
                    "Échec de synchronisation d'au moins un checkpoint "
                    "intermédiaire vers Hugging Face — voir logs "
                    "ColabCheckpointSync ci-dessus pour le détail."
                )
        else:
            logger.info(
                "Aucun checkpoint intermédiaire local (dossier "
                "'checkpoint-*' absent) — rien à synchroniser à ce niveau."
            )

        # 2) Modèle final (celui produit par trainer.save_model() ci-dessus)
        #    -> publié sous sft-final/ (PAS la racine), PUIS supprimé
        #    localement.
        #
        # FIX HUB-COLLISION — la stratégie de déploiement confirme qu'un
        # Hugging Face Space API (RemDev-AI/medical-triage-agent-ai-poc-api)
        # charge directement la RACINE de hub_model_id en production.
        # Cette racine doit donc être réservée au modèle FINAL DPO
        # (train_dpo.py), pas au modèle SFT intermédiaire. Avant ce
        # correctif, les deux scripts poussaient vers la racine et le
        # second (DPO, exécuté après SFT dans le pipeline) écrasait
        # silencieusement les fichiers du premier. sft-final/ conserve
        # une trace du modèle SFT pur, exigée par la mission
        # ("garder une traçabilité... checkpoints pour la reprise de
        # l'entraînement").
        SFT_FINAL_MODEL_REMOTE_PREFIX = "sft-final"

        logger.info(
            "Publication du modèle final SFT sur Hugging Face "
            "(hub_model_id=%s, chemin=%s/)...",
            CONFIG["model"]["hub_model_id"],
            SFT_FINAL_MODEL_REMOTE_PREFIX,
        )
        from huggingface_hub import HfApi

        output_dir_path = Path(CONFIG["training"]["output_dir"])
        hf_api = HfApi()

        # FIX HUB-6 — output_dir peut contenir un sous-dossier ".cache/"
        # (cache interne de huggingface_hub, ex: HF_HOME/HUGGINGFACE_HUB_CACHE
        # pointant par erreur à l'intérieur de output_dir, ou résidu d'un
        # snapshot_download() antérieur). Ce sous-dossier ne fait PAS partie
        # du modèle et est de toute façon exclu par upload_folder() (fichiers
        # de cache/metadata internes, jamais poussés sur le Hub). Il doit
        # donc être ignoré à l'upload ET lors de la vérification post-upload,
        # sous peine de faux positifs dans missing_files (cf. RuntimeError
        # précédent : 14 fichiers ".cache/huggingface/..." signalés à tort
        # comme "manquants" alors qu'ils n'ont jamais dû être uploadés).
        IGNORED_LOCAL_DIRS = (".cache",)

        try:
            hf_api.upload_folder(
                folder_path=str(output_dir_path),
                repo_id=CONFIG["model"]["hub_model_id"],
                repo_type="model",
                path_in_repo=SFT_FINAL_MODEL_REMOTE_PREFIX,
                commit_message="SFT validation run — final model",
                ignore_patterns=[
                    ".cache/**",
                    ".cache",
                    "*.metadata",
                    "CACHEDIR.TAG",
                    ".gitignore",
                ],
            )
            logger.info("Modèle final SFT publié sur Hugging Face.")

            # FIX HUB-3 — vérification explicite post-upload : on ne fait
            # pas confiance au seul "succès" de upload_folder() (qui peut
            # masquer un upload partiel). On relit la liste des fichiers
            # réellement présents sur le Hub et on la compare à la liste
            # des fichiers locaux attendus AVANT toute suppression locale.
            #
            # FIX HUB-6 — on exclut ici les mêmes fichiers/dossiers que ceux
            # passés à ignore_patterns ci-dessus, pour que la vérification
            # compare des ensembles cohérents (fichiers réellement candidats
            # à l'upload, pas le cache technique local).
            local_files = {
                str(path.relative_to(output_dir_path)).replace("\\", "/")
                for path in output_dir_path.rglob("*")
                if path.is_file()
                and not any(
                    part in IGNORED_LOCAL_DIRS
                    for part in path.relative_to(output_dir_path).parts
                )
            }

            remote_prefix = f"{SFT_FINAL_MODEL_REMOTE_PREFIX}/"
            remote_files = {
                file[len(remote_prefix) :]
                for file in hf_api.list_repo_files(
                    repo_id=CONFIG["model"]["hub_model_id"],
                    repo_type="model",
                )
                if file.startswith(remote_prefix)
            }

            missing_files = local_files - remote_files

            if missing_files:
                raise RuntimeError(
                    f"Vérification post-upload échouée : {len(missing_files)} "
                    f"fichier(s) local(aux) absent(s) du Hub après upload_folder() : "  # noqa: E501
                    f"{sorted(missing_files)}"
                )

            logger.info(
                "Vérification post-upload OK : %d fichier(s) confirmé(s) "
                "présent(s) sur %s/%s.",
                len(local_files),
                CONFIG["model"]["hub_model_id"],
                SFT_FINAL_MODEL_REMOTE_PREFIX,
            )

            if checkpoint_sync.cleanup_after_upload:
                import shutil

                shutil.rmtree(output_dir_path, ignore_errors=True)
                logger.info(
                    "Répertoire local %s supprimé après publication réussie "
                    "et vérifiée (aucune persistance locale sur Colab).",
                    output_dir_path,
                )
        except Exception:
            logger.exception(
                "Échec de la publication ou de la vérification du modèle "
                "final sur Hugging Face — le répertoire local %s est "
                "CONSERVÉ pour éviter toute perte de données.",
                output_dir_path,
            )
            raise
    else:
        logger.info(
            "publish_to_hf=False — modèle conservé en local uniquement "
            "(%s), aucun push vers le Hub.",
            CONFIG["training"]["output_dir"],
        )

    TrainingUtils.finalize_wandb_run(wandb_run)

    logger.info("SFT validation run completed.")


# if __name__ == "__main__":
#     train()

if __name__ == "__main__":
    train(publish_to_hf=True)
