"""
diagnose_lm_head_merge.py

Objectif
--------
Vérifier, de façon définitive, l'hypothèse suivante :

    "Le checkpoint adapter (checkpoints/dpo/checkpoint-dpo-32) n'a
    JAMAIS réellement sauvegardé de poids lm_head distinct dans
    adapter_model.safetensors (bug connu de PEFT avec
    tie_word_embeddings=True, cf. peft#2777), donc
    merge_and_unload() n'avait rien à fusionner pour lm_head, et le
    modèle 'fusionné' publié sur le Hub a en réalité un lm_head
    IDENTIQUE à celui du modèle de base Qwen3-1.7B-Base — jamais
    fine-tuné."

Si cette hypothèse est confirmée, elle explique les dérives
observées en prod (générations qui partent dans des langues/formats
non pertinents, arrêt erratique, échecs de format sur /triage/) :
la tête de sortie du modèle ne reflète pas le fine-tuning DPO,
malgré une fusion LoRA par ailleurs correcte sur les autres couches.

Ce script fait DEUX vérifications indépendantes, de la plus légère
à la plus définitive :

  ÉTAPE 1 (légère, quelques secondes, pas de GPU) :
      Liste EXHAUSTIVE des clés présentes dans
      adapter_model.safetensors du checkpoint adapter. Confirme ou
      infirme directement l'absence de toute clé modules_to_save
      pour lm_head/embed_tokens.

  ÉTAPE 2 (plus lourde, télécharge les 2 modèles) :
      Compare tenseur par tenseur le poids lm_head du modèle publié
      RemDev-AI/medical-triage-agent-ai-poc-merged avec celui du
      modèle de base Qwen/Qwen3-1.7B-Base. S'ils sont identiques (ou
      quasi-identiques, à la tolérance du cast bf16 près), c'est la
      confirmation définitive que le lm_head n'a jamais été
      fine-tuné dans le modèle publié.

Usage
-----
    pip install torch transformers safetensors huggingface_hub --quiet
    python scripts/diagnose_lm_head_merge.py

Variables à ajuster si besoin (adapter_repo_id, subfolder, base
model, modèle fusionné) sont regroupées en haut de fichier.
"""

from __future__ import annotations

import sys

# =========================================================
# Configuration — à ajuster si les repos/checkpoints changent
# =========================================================

BASE_MODEL_NAME = "Qwen/Qwen3-1.7B-Base"
ADAPTER_REPO_ID = "RemDev-AI/medical-triage-agent-ai-poc-models"
ADAPTER_SUBFOLDER = "checkpoints/dpo/checkpoint-dpo-32"
MERGED_MODEL_NAME = "RemDev-AI/medical-triage-agent-ai-poc-merged"

# Tolérance pour la comparaison de tenseurs en bfloat16 (le cast
# fp32 -> bf16 lors de la sauvegarde du modèle fusionné introduit un
# arrondi ; on tolère un écart minime plutôt que d'exiger une
# égalité bit-à-bit, qui donnerait un faux négatif même si le
# lm_head N'A PAS été fine-tuné).
ATOL = 1e-2
RTOL = 1e-2


def step1_list_adapter_keys() -> list[str]:
    """
    Télécharge adapter_model.safetensors (ou .bin en repli) et liste
    TOUTES les clés qu'il contient, sans aucun filtre sur le nom
    (contrairement à get_adapter_lm_head_target_size() dans
    merge_lora_adapter.py, qui cherche un pattern précis — ici on
    veut voir l'exhaustivité brute pour être certain de ne rien
    manquer).
    """

    print("=" * 70)
    print("ÉTAPE 1 — Clés présentes dans adapter_model.safetensors")
    print("=" * 70)

    from huggingface_hub import hf_hub_download

    candidate_filenames = [
        "adapter_model.safetensors",
        "adapter_model.bin",
    ]

    all_keys: list[str] = []

    for filename in candidate_filenames:

        try:
            weight_path = hf_hub_download(
                repo_id=ADAPTER_REPO_ID,
                subfolder=ADAPTER_SUBFOLDER,
                filename=filename,
            )
        except Exception as exc:
            print(f"  (introuvable : {filename} — {exc})")
            continue

        print(f"\nFichier trouvé : {filename}")

        if filename.endswith(".safetensors"):
            from safetensors import safe_open

            with safe_open(weight_path, framework="pt") as f:
                all_keys = list(f.keys())
        else:
            import torch as _torch

            state_dict = _torch.load(weight_path, map_location="cpu")
            all_keys = list(state_dict.keys())

        break

    if not all_keys:
        print("\n⚠️  Aucun fichier de poids d'adapter trouvé — vérifie "
              "ADAPTER_REPO_ID / ADAPTER_SUBFOLDER.")
        return []

    print(f"\nNombre total de clés : {len(all_keys)}\n")

    lora_keys = [k for k in all_keys if "lora_" in k]
    modules_to_save_keys = [k for k in all_keys if "modules_to_save" in k]
    lm_head_keys = [k for k in all_keys if "lm_head" in k]
    embed_keys = [k for k in all_keys if "embed_tokens" in k or "wte" in k]
    other_keys = [
        k
        for k in all_keys
        if k not in lora_keys
        and k not in modules_to_save_keys
        and k not in lm_head_keys
        and k not in embed_keys
    ]

    print(f"Clés LoRA (lora_A/lora_B)        : {len(lora_keys)}")
    print(f"Clés modules_to_save              : {len(modules_to_save_keys)}")
    for k in modules_to_save_keys:
        print(f"    - {k}")
    print(f"Clés contenant 'lm_head'          : {len(lm_head_keys)}")
    for k in lm_head_keys:
        print(f"    - {k}")
    print(f"Clés contenant 'embed_tokens'/'wte': {len(embed_keys)}")
    for k in embed_keys:
        print(f"    - {k}")
    print(f"Autres clés non catégorisées      : {len(other_keys)}")
    for k in other_keys[:20]:
        print(f"    - {k}")
    if len(other_keys) > 20:
        print(f"    ... (+{len(other_keys) - 20} autres)")

    if not modules_to_save_keys:
        print(
            "\n🔴 CONFIRMÉ : aucune clé 'modules_to_save' dans "
            "adapter_model.safetensors — le lm_head fine-tuné n'a "
            "JAMAIS été sauvegardé par PEFT pour ce checkpoint. "
            "merge_and_unload() n'avait donc rien à fusionner pour "
            "lm_head."
        )
    else:
        print(
            "\n🟢 Des clés 'modules_to_save' existent bel et bien — "
            "l'hypothèse d'une absence totale est infirmée. Passe à "
            "l'étape 2 pour vérifier si elles ont néanmoins été "
            "correctement appliquées lors du merge."
        )

    return all_keys


def step2_compare_lm_head_weights() -> None:
    """
    Compare le lm_head du modèle PUBLIÉ (fusionné) avec celui du
    modèle de BASE. Vérification définitive et indépendante de
    l'étape 1 : même si une clé modules_to_save existait (étape 1
    positive), il faut confirmer qu'elle a effectivement changé la
    valeur du poids dans le modèle final réellement servi en prod.
    """

    print("\n" + "=" * 70)
    print("ÉTAPE 2 — Comparaison lm_head : modèle fusionné vs modèle de base")
    print("=" * 70)

    import torch
    from transformers import AutoModelForCausalLM

    print(f"\nChargement du lm_head du modèle de base ({BASE_MODEL_NAME})...")
    base_model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL_NAME,
        dtype=torch.float32,
    )
    base_lm_head = base_model.get_output_embeddings().weight.detach().clone()
    del base_model

    print(f"Chargement du lm_head du modèle fusionné ({MERGED_MODEL_NAME})...")
    merged_model = AutoModelForCausalLM.from_pretrained(
        MERGED_MODEL_NAME,
        dtype=torch.float32,
    )
    merged_lm_head = merged_model.get_output_embeddings().weight.detach().clone()
    del merged_model

    print(f"\nShape lm_head base   : {tuple(base_lm_head.shape)}")
    print(f"Shape lm_head fusionné: {tuple(merged_lm_head.shape)}")

    if base_lm_head.shape != merged_lm_head.shape:
        print(
            "\n⚠️  Les deux tenseurs n'ont pas la même shape — "
            "comparaison directe impossible. Un resize a "
            "manifestement eu lieu (ou un décalage de version). "
            "Vérifie manuellement."
        )
        return

    are_close = torch.allclose(
        base_lm_head, merged_lm_head, atol=ATOL, rtol=RTOL
    )
    max_abs_diff = (base_lm_head - merged_lm_head).abs().max().item()
    mean_abs_diff = (base_lm_head - merged_lm_head).abs().mean().item()

    print(f"\nÉcart absolu maximal  : {max_abs_diff:.6f}")
    print(f"Écart absolu moyen    : {mean_abs_diff:.6f}")
    print(f"Tolérance utilisée    : atol={ATOL}, rtol={RTOL}")

    if are_close:
        print(
            "\n🔴 CONFIRMÉ : le lm_head du modèle fusionné est "
            "QUASI-IDENTIQUE à celui du modèle de base non fine-tuné. "
            "Le fine-tuning DPO n'a jamais atteint la tête de sortie "
            "réellement servie en production. C'est très probablement "
            "la cause racine des dérives de génération observées "
            "(langues/formats hors-sujet, arrêt erratique)."
        )
        print(
            "\nAction recommandée : ce n'est plus un problème de "
            "merge_lora_adapter.py — le correctif doit se faire "
            "AVANT/PENDANT l'entraînement (SFT/DPO), en s'assurant "
            "que l'entraînement PEFT gère correctement le cas "
            "tie_word_embeddings=True (ex: untie explicitement les "
            "embeddings avant l'entraînement, ou vérifier la version "
            "de peft utilisée à l'entraînement et son support de "
            "ensure_weight_tying dès la phase d'entraînement, pas "
            "seulement au merge)."
        )
    else:
        print(
            "\n🟢 Les deux lm_head sont significativement différents — "
            "le fine-tuning a bien atteint la tête de sortie. "
            "L'hypothèse est infirmée : la cause des dérives observées "
            "est probablement ailleurs (qualité intrinsèque du "
            "fine-tuning, paramètres de sampling, taille du modèle "
            "1.7B, etc.) plutôt qu'un défaut de merge."
        )


def main() -> int:

    all_keys = step1_list_adapter_keys()

    print(
        "\nPasser à l'étape 2 (téléchargement de 2 modèles complets, "
        "peut prendre plusieurs minutes et quelques Go de disque) ?"
    )
    answer = input("Continuer ? [y/N] : ").strip().lower()

    if answer != "y":
        print("Arrêt après l'étape 1.")
        return 0

    step2_compare_lm_head_weights()

    return 0


if __name__ == "__main__":
    sys.exit(main())
