# medical-triage-agent-ai-poc/backend/app/llm/inference/vllm_engine.py

"""
Moteur d'inférence vLLM (in-process).

Introduit à l'étape 3 pour répondre au point 2 du
cahier des charges :

    "Conteneuriser l'application avec Docker et
    l'exposer via une API (FastAPI) en utilisant
    vLLM pour une inférence optimisée."

Approche retenue : intégration en process via
AsyncLLMEngine (pas de serveur HTTP vLLM séparé).
L'adaptateur LoRA chargé est la POLICY FINALE post-DPO
(fichiers à la racine de
checkpoints/dpo/checkpoint-dpo-32/), pas le sous-dossier
"ref/" qui correspond au modèle de référence figé utilisé
uniquement pour le calcul de la divergence KL pendant
l'entraînement DPO — jamais un artefact de production.

Le téléchargement de l'adaptateur est filtré (allow_patterns)
pour ne récupérer que les fichiers utiles à l'inférence
(adapter_config.json, adapter_model.safetensors, tokenizer*),
et exclut explicitement les états d'entraînement
(optimizer.pt, scheduler.pt, scaler.pt, rng_state.pth,
training_args.bin, trainer_state.json) qui représentent
l'essentiel du volume de chaque checkpoint (~3+ Go) et sont
inutiles à l'inférence.

IMPORTANT — BACKENDS SUPPORTÉS (GPU CUDA et CPU) :
Ce module supporte deux backends vLLM :

- GPU CUDA (backend standard) : utilisé automatiquement si
  torch.cuda.is_available() est True.
- CPU natif (roues pré-compilées vLLM CPU, cf.
  requirements.txt "OPTION B" et Dockerfile associé) : utilisé
  en l'absence de GPU CUDA, à condition que le backend CPU de
  vLLM soit explicitement configuré via les variables
  d'environnement dédiées (VLLM_CPU_KVCACHE_SPACE,
  VLLM_CPU_OMP_THREADS_BIND — cf. Dockerfile). C'est le cas
  d'usage de ce POC pédagogique, déployé sur un tier Hugging
  Face Space "CPU basic" (offre gratuite, sans GPU).

Si ni CUDA ni le backend CPU de vLLM ne sont configurés (ex :
environnement de développement local sans réglage particulier,
ou CI sans GPU et sans intention d'utiliser vLLM), un garde-fou
explicite est levé ci-dessous plutôt que de laisser vLLM échouer
silencieusement/confusément. Dans ce dernier cas, utiliser
USE_VLLM=false et le chemin d'inférence CPU historique
(app/llm/loaders/model_loader.py) à la place.
"""

from __future__ import annotations

import logging
import os
import threading
from pathlib import Path
from typing import Optional

from app.deployment.huggingface.hf_space_runtime import (
    runtime_config,
)
from app.llm.inference.generate import (
    clean_response,
    _build_chat_prompt,
)

logger = logging.getLogger(__name__)

_engine_lock = threading.Lock()
_engine_instance = None
_LORA_ADAPTER_NAME = "medical-triage-lora"

# Modèle de base utilisé pour le fine-tuning
# (cf. backend/app/training/sft/sft_config_validation.yaml,
# clé model.base_model). Il s'agit d'un modèle public
# Hugging Face, distinct du dépôt de résultats
# RemDev-AI/medical-triage-agent-ai-poc-models (qui ne
# contient QUE des adaptateurs LoRA, jamais les poids
# complets du modèle de base).
_BASE_MODEL_NAME = "Qwen/Qwen3-1.7B-Base"

# Sous-dossier du Hub contenant l'adaptateur LoRA final
# (policy post-DPO). NE PAS pointer vers "ref/" : voir
# docstring du module.
_ADAPTER_SUBFOLDER = "checkpoints/dpo/checkpoint-dpo-32"

# Fichiers strictement nécessaires à l'inférence pour cet
# adaptateur. Exclut volontairement optimizer.pt,
# scheduler.pt, scaler.pt, rng_state.pth, training_args.bin,
# trainer_state.json (états d'entraînement, ~3+ Go/checkpoint,
# jamais utiles au serving).
_ADAPTER_ALLOW_PATTERNS = [
    f"{_ADAPTER_SUBFOLDER}/adapter_config.json",
    f"{_ADAPTER_SUBFOLDER}/adapter_model.safetensors",
    f"{_ADAPTER_SUBFOLDER}/tokenizer.json",
    f"{_ADAPTER_SUBFOLDER}/tokenizer_config.json",
    f"{_ADAPTER_SUBFOLDER}/chat_template.jinja",
]

# Répertoire local de cache pour l'adaptateur téléchargé.
# HF_HOME est déjà positionné sur /tmp/huggingface dans
# Dockerfile.hf ; on isole l'adaptateur dans un sous-dossier
# dédié pour rester explicite sur ce qui est téléchargé ici.
_ADAPTER_LOCAL_CACHE_DIR = Path(
    os.getenv("ADAPTER_LOCAL_CACHE_DIR", "/tmp/medical-triage-adapter")
)

# Variables d'environnement qui, si présentes, signalent que le
# backend CPU natif de vLLM a été explicitement configuré (cf.
# Dockerfile "OPTION B" : ENV VLLM_CPU_KVCACHE_SPACE=4,
# ENV VLLM_CPU_OMP_THREADS_BIND=0). Leur présence sert de
# condition explicite pour autoriser vLLM à démarrer sans GPU
# CUDA, plutôt que de se fier à une simple absence de CUDA
# (qui pourrait aussi signifier "vLLM n'a pas du tout été prévu
# ici", ex : environnement de dev local ou CI).
_VLLM_CPU_BACKEND_ENV_MARKERS = (
    "VLLM_CPU_KVCACHE_SPACE",
    "VLLM_CPU_OMP_THREADS_BIND",
)


def _resolve_adapter_repository() -> str:
    """
    Retourne le repo_id Hugging Face (avec namespace) contenant
    l'adaptateur LoRA final.

    Le sous-dossier (checkpoints/dpo/checkpoint-dpo-32) est géré
    séparément via _ADAPTER_SUBFOLDER / _ADAPTER_ALLOW_PATTERNS,
    car un repo_id Hugging Face ne peut pas inclure de sous-chemin
    collé (ex: "org/repo/sous-dossier" n'est PAS un repo_id valide).

    runtime_config.model_repository doit inclure le namespace
    complet, ex: "RemDev-AI/medical-triage-agent-ai-poc-models"
    (cf. .env : HF_MODEL_REPOSITORY).
    """

    repo_id = runtime_config.model_repository

    if "/" not in repo_id:
        raise ValueError(
            "HF_MODEL_REPOSITORY doit inclure le namespace complet "
            f"(ex: 'RemDev-AI/medical-triage-agent-ai-poc-models'), "
            f"valeur actuelle sans namespace : '{repo_id}'."
        )

    return repo_id


def _ensure_adapter_downloaded() -> str:
    """
    Télécharge (si nécessaire) uniquement les fichiers d'inférence
    de l'adaptateur LoRA final, et retourne le chemin local du
    dossier contenant l'adaptateur (compatible LoRARequest, qui
    attend un chemin filesystem, pas un repo_id + sous-dossier).
    """

    from huggingface_hub import snapshot_download

    repo_id = _resolve_adapter_repository()

    logger.info(
        "Downloading inference-only adapter files from %s/%s "
        "(training state files excluded)",
        repo_id,
        _ADAPTER_SUBFOLDER,
    )

    local_snapshot_dir = snapshot_download(
        repo_id=repo_id,
        allow_patterns=_ADAPTER_ALLOW_PATTERNS,
        local_dir=str(_ADAPTER_LOCAL_CACHE_DIR),
    )

    adapter_local_path = Path(local_snapshot_dir) / _ADAPTER_SUBFOLDER

    if not (adapter_local_path / "adapter_model.safetensors").exists():
        raise FileNotFoundError(
            "adapter_model.safetensors introuvable après téléchargement "
            f"filtré dans {adapter_local_path}. Vérifiez "
            "_ADAPTER_SUBFOLDER et _ADAPTER_ALLOW_PATTERNS."
        )

    logger.info("Adapter ready at %s", adapter_local_path)

    return str(adapter_local_path)


def _is_vllm_cpu_backend_configured() -> bool:
    """
    Indique si le backend CPU natif de vLLM a été explicitement
    configuré dans cet environnement (cf. Dockerfile "OPTION B").

    On se base sur la présence des variables d'environnement que
    seul ce backend consomme, plutôt que sur l'absence de CUDA
    seule, pour distinguer :
    - "pas de GPU, mais vLLM CPU explicitement préparé"
      (ce POC, Hugging Face Space CPU basic)
    - "pas de GPU, et vLLM n'a jamais été configuré ici"
      (dev local, CI sans USE_VLLM)
    """

    return any(
        os.getenv(marker) is not None for marker in _VLLM_CPU_BACKEND_ENV_MARKERS
    )


def _assert_inference_backend_available() -> None:
    """
    Garde-fou : vLLM doit tourner soit sur GPU CUDA, soit sur son
    backend CPU natif explicitement configuré. Si ni l'un ni
    l'autre n'est disponible, on préfère un échec immédiat et
    clair plutôt qu'un comportement dégradé ou une erreur confuse
    plus tard dans vLLM.

    (Anciennement `_assert_gpu_available`, qui exigeait

    inconditionnellement un GPU CUDA — trop strict pour ce POC,
    qui utilise volontairement les roues CPU natives de vLLM
    (cf. requirements.txt "OPTION B", Dockerfile) sur le tier
    gratuit d'un Hugging Face Space, sans GPU.)
    """

    import torch

    if torch.cuda.is_available():
        return

    if _is_vllm_cpu_backend_configured():
        logger.info(
            "No CUDA GPU detected, but vLLM CPU backend is explicitly "
            "configured (%s) — proceeding on CPU.",
            ", ".join(
                m for m in _VLLM_CPU_BACKEND_ENV_MARKERS if os.getenv(m) is not None
            ),
        )
        return

    raise RuntimeError(
        "vLLM engine requested but no CUDA GPU is available, and the "
        "vLLM CPU backend does not appear to be configured in this "
        "environment (none of "
        f"{_VLLM_CPU_BACKEND_ENV_MARKERS} is set). On a CPU-only "
        "Hugging Face Space tier, either configure the vLLM CPU "
        "backend (see Dockerfile 'OPTION B') or set USE_VLLM=false to "
        "rely on the CPU inference path "
        "(app/llm/loaders/model_loader.py) instead."
    )


def get_vllm_engine():
    """
    Retourne l'instance singleton d'AsyncLLMEngine,
    en la créant si nécessaire (lazy init, thread-safe).

    Le chargement est volontairement paresseux : il ne
    doit être déclenché qu'au premier appel réel de
    génération, afin de ne jamais impacter le démarrage
    de l'API (health checks, /docs, tests unitaires) ni
    l'exécution de la CI (aucun GPU disponible).
    """

    global _engine_instance

    if _engine_instance is not None:
        return _engine_instance

    with _engine_lock:

        if _engine_instance is not None:
            return _engine_instance

        try:
            from vllm import AsyncEngineArgs
            from vllm import AsyncLLMEngine
            from vllm.lora.request import LoRARequest  # noqa : F401
        except ImportError as exc:
            raise RuntimeError(
                "vLLM n'est pas installé ou "
                "indisponible dans cet environnement. "
                "Vérifiez requirements.txt et la "
                "présence d'un GPU compatible, ou du "
                "backend CPU natif de vLLM."
            ) from exc

        _assert_inference_backend_available()

        import torch

        is_cpu_backend = not torch.cuda.is_available()

        engine_kwargs = dict(
            model=_BASE_MODEL_NAME,
            enable_lora=True,
            max_lora_rank=64,
            dtype="bfloat16",
            max_model_len=(
                runtime_config.max_input_tokens + runtime_config.max_output_tokens
            ),
            trust_remote_code=True,
        )

        if is_cpu_backend:
            # gpu_memory_utilization n'a pas de sens sur le backend
            # CPU (pas de VRAM à réserver) : le dimensionnement du
            # KV cache est piloté par la variable d'environnement
            # VLLM_CPU_KVCACHE_SPACE (cf. Dockerfile), pas par un
            # paramètre d'AsyncEngineArgs.
            engine_kwargs["device"] = "cpu"
        else:
            engine_kwargs["gpu_memory_utilization"] = 0.85

        engine_args = AsyncEngineArgs(**engine_kwargs)

        logger.info(
            "Initializing vLLM AsyncLLMEngine "
            "(backend=%s, base_model=%s, adapter_repo=%s, "
            "adapter_subfolder=%s, enable_lora=True)",
            "cpu" if is_cpu_backend else "cuda",
            _BASE_MODEL_NAME,
            _resolve_adapter_repository(),
            _ADAPTER_SUBFOLDER,
        )

        _engine_instance = AsyncLLMEngine.from_engine_args(engine_args)

        logger.info("vLLM AsyncLLMEngine initialized.")

        return _engine_instance


async def generate_response_vllm(
    system_prompt: str,
    user_prompt: str,
    max_new_tokens: int = 256,
    temperature: float = 0.2,
    top_p: float = 0.9,
    repetition_penalty: float = 1.1,
    request_id: Optional[str] = None,
) -> str:
    """
    Génère une réponse via vLLM (AsyncLLMEngine),
    en utilisant l'adaptateur LoRA final (policy post-DPO)
    chargé dynamiquement (--enable-lora), sans merge.

    Signature volontairement alignée sur
    generate.generate_response() (mêmes paramètres
    de sampling) afin de permettre un dispatch
    transparent depuis TriageEngine, sans changer
    l'API publique existante.
    """

    from vllm import SamplingParams
    from vllm.lora.request import LoRARequest
    import uuid

    engine = get_vllm_engine()

    adapter_local_path = _ensure_adapter_downloaded()

    prompt = _build_chat_prompt(
        system_prompt=system_prompt,
        user_prompt=user_prompt,
    )

    sampling_params = SamplingParams(
        max_tokens=max_new_tokens,
        temperature=temperature,
        top_p=top_p,
        repetition_penalty=repetition_penalty,
    )

    lora_request = LoRARequest(
        _LORA_ADAPTER_NAME,
        1,
        adapter_local_path,
    )

    resolved_request_id = request_id or str(uuid.uuid4())

    final_output = None

    async for output in engine.generate(
        prompt=prompt,
        sampling_params=sampling_params,
        request_id=resolved_request_id,
        lora_request=lora_request,
    ):
        final_output = output

    if final_output is None or not final_output.outputs:
        raise RuntimeError(
            "vLLM did not return any output for " f"request_id={resolved_request_id}"
        )

    raw_text = final_output.outputs[0].text

    return clean_response(raw_text)


def is_vllm_enabled() -> bool:
    """
    Point de vérité unique pour savoir si le
    moteur vLLM doit être utilisé, piloté par
    la variable d'environnement USE_VLLM
    (cf. hf_space_runtime.py).
    """

    return runtime_config.use_vllm
