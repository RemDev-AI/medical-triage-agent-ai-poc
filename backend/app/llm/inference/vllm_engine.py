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

CORRECTIF (2026-07-20) — MERGE OFFLINE AU LIEU DE LoRA DYNAMIQUE
------------------------------------------------------------------
Ce module chargeait auparavant le modèle de base
(Qwen/Qwen3-1.7B-Base) puis l'adaptateur LoRA final post-DPO
(checkpoints/dpo/checkpoint-dpo-32) dynamiquement via
`enable_lora=True` + `LoRARequest`.

Cette approche a été abandonnée car incompatible avec vLLM :
l'adapter_config.json de ce checkpoint contient
`"modules_to_save": ["lm_head"]` (lm_head entièrement fine-tuné,
hors LoRA — probablement dû à des tokens spéciaux ajoutés avant
le SFT/DPO). Or vLLM ne supporte pas ce mécanisme PEFT lors du
chargement dynamique d'un adapter :

    RuntimeError: Worker failed with error
    'vLLM only supports modules_to_save being None.'

Cette erreur ne se déclenchait pas au démarrage (le moteur
s'initialisait normalement avec le seul modèle de base), mais au
tout premier appel de génération, provoquant un `EngineDeadError`
qui tuait définitivement l'EngineCore (toutes les requêtes
suivantes échouaient, y compris sur des routes non concernées).

Le correctif consiste à fusionner l'adaptateur (LoRA + lm_head)
dans le modèle de base UNE FOIS, hors ligne
(cf. scripts/merge_lora_adapter.py), et à publier le résultat sur
un repo Hugging Face dédié (_MERGED_MODEL_NAME ci-dessous). Ce
module sert désormais directement ce modèle fusionné, sans aucune
dépendance à PEFT/LoRA au runtime :

- Plus de `enable_lora`, plus de `LoRARequest`, plus de
  téléchargement/filtrage d'adapter au démarrage du serveur.
- Le modèle fusionné se comporte comme un modèle "normal" pour
  vLLM : plus aucun risque lié à modules_to_save.
- Contrepartie assumée : impossible de hot-swap plusieurs
  adaptateurs sur ce serveur. Non pertinent ici (un seul
  adaptateur, la policy finale post-DPO, jamais plusieurs
  variantes servies en parallèle sur ce POC).

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
from typing import Optional

from app.deployment.huggingface.hf_space_runtime import (
    runtime_config,
)
from app.llm.inference.generate import (
    clean_response,
    build_chat_prompt_with_tokenizer,
)

logger = logging.getLogger(__name__)

_engine_lock = threading.Lock()
_engine_instance = None

# Tokenizer chargé séparément du moteur vLLM. AsyncLLMEngine charge
# bien SA PROPRE copie interne du tokenizer pour la tokenisation, mais
# ne l'expose pas publiquement pour construire un prompt via
# apply_chat_template — d'où ce singleton dédié, léger (le tokenizer
# seul, sans les poids du modèle), chargé une fois via
# transformers.AutoTokenizer pour accéder au vrai chat_template.jinja
# publié avec _MERGED_MODEL_NAME. Introduit suite à un bug de dérive
# de génération causé par un gabarit ChatML fait main désynchronisé
# du format d'entraînement réel (cf. generate.py,
# build_chat_prompt_with_tokenizer).
_tokenizer_lock = threading.Lock()
_tokenizer_instance = None

# Modèle fusionné (base + adaptateur LoRA final post-DPO +
# modules_to_save["lm_head"]), produit une seule fois hors ligne
# par scripts/merge_lora_adapter.py et publié sur ce repo dédié.
#
# NE PLUS pointer vers le modèle de base nu (Qwen/Qwen3-1.7B-Base) :
# ce module ne charge plus aucun adaptateur au runtime, servir la
# base seule reviendrait à servir un modèle non fine-tuné.
#
# Remplacer par le repo réel une fois le merge effectué et poussé,
# ex: "RemDev-AI/medical-triage-agent-ai-poc-merged". Surchargeable
# via la variable d'environnement dédiée pour permettre de changer
# de version fusionnée sans modifier le code.
_MERGED_MODEL_NAME = os.getenv(
    "MERGED_MODEL_REPOSITORY",
    "RemDev-AI/medical-triage-agent-ai-poc-merged",
)

# Révision (commit SHA ou tag) du repo ci-dessus à charger.
# Par défaut "main" pour ne pas casser le POC si aucune valeur n'est
# fournie, mais fortement recommandé de surcharger cette variable
# avec un SHA de commit figé une fois le modèle publié, pour éviter
# qu'un futur push sur le repo HF ne change silencieusement le
# modèle/tokenizer servi (cf. Bandit B615 / CWE-494).
_MERGED_MODEL_REVISION = os.getenv("MERGED_MODEL_REVISION", "main")

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


def get_vllm_tokenizer():
    """
    Retourne l'instance singleton du tokenizer associé à
    _MERGED_MODEL_NAME, en la créant si nécessaire (lazy init,
    thread-safe — même pattern que get_vllm_engine()).

    Utilisé pour construire les prompts via le vrai chat_template du
    modèle (cf. generate.build_chat_prompt_with_tokenizer), plutôt
    que via un gabarit ChatML fait main potentiellement désynchronisé
    du format d'entraînement réel.

    Chargement volontairement séparé de get_vllm_engine() : le
    tokenizer seul est nécessaire dès la construction du prompt,
    avant même d'appeler engine.generate(), et ne doit pas dépendre
    du chargement complet (lourd) du moteur vLLM.
    """

    global _tokenizer_instance

    if _tokenizer_instance is not None:
        return _tokenizer_instance

    with _tokenizer_lock:

        if _tokenizer_instance is not None:
            return _tokenizer_instance

        from transformers import AutoTokenizer

        logger.info(
            "Loading tokenizer for chat template resolution (model=%s)",
            _MERGED_MODEL_NAME,
        )

        _tokenizer_instance = AutoTokenizer.from_pretrained(
            _MERGED_MODEL_NAME,
            revision=_MERGED_MODEL_REVISION,
            trust_remote_code=True,
        )

        logger.info("Tokenizer loaded.")

        return _tokenizer_instance


def get_vllm_engine():
    """
    Retourne l'instance singleton d'AsyncLLMEngine,
    en la créant si nécessaire (lazy init, thread-safe).

    Le chargement est volontairement paresseux : il ne
    doit être déclenché qu'au premier appel réel de
    génération, afin de ne jamais impacter le démarrage
    de l'API (health checks, /docs, tests unitaires) ni
    l'exécution de la CI (aucun GPU disponible).

    Sert directement le modèle fusionné (_MERGED_MODEL_NAME) :
    plus de LoRA dynamique, plus de dépendance PEFT au runtime
    (cf. docstring du module pour le détail du correctif).
    """

    global _engine_instance

    if _engine_instance is not None:
        return _engine_instance

    with _engine_lock:

        if _engine_instance is not None:
            return _engine_instance

        import torch

        is_cpu_backend = not torch.cuda.is_available()

        if is_cpu_backend:
            # NOTE : une tentative précédente forçait ici
            # os.environ["VLLM_TARGET_DEVICE"] = "cpu" avant l'import
            # de vllm. Cela s'est révélé inefficace et a été retiré :
            # VLLM_TARGET_DEVICE est un réglage de COMPILATION du
            # wheel (cf. vllm/setup.py), pas une variable lue au
            # runtime. La détection de plateforme
            # (vllm/platforms/__init__.py, cpu_platform_plugin())
            # décide de current_platform en inspectant la CHAÎNE DE
            # VERSION du paquet vllm réellement installé : elle doit
            # contenir "cpu" (ex: "0.25.1+cpu"), suffixe présent
            # uniquement sur les wheels compilés pour CPU (cf.
            # requirements.txt : "vllm==0.25.1+cpu", pas juste
            # "vllm==0.25.1"). Sans ce suffixe, current_platform ne
            # résout aucune plateforme et create_engine_config() lève
            # "RuntimeError: Device string must not be empty" (bug
            # observé en prod, root-caused à ce niveau).
            #
            # Garde-fou : on vérifie ici que le wheel installé porte
            # bien ce suffixe, pour échouer avec un message clair
            # plutôt que de laisser vLLM lever une erreur pydantic
            # confuse plus loin si requirements.txt venait à être
            # mal résolu à nouveau (ex: ambiguïté entre plusieurs
            # --extra-index-url).
            from importlib.metadata import version as _pkg_version

            installed_vllm_version = _pkg_version("vllm")

            if "cpu" not in installed_vllm_version.lower():
                raise RuntimeError(
                    "Backend CPU attendu, mais le wheel vLLM installé "
                    f"('{installed_vllm_version}') ne porte pas le "
                    "suffixe '+cpu' : ce n'est pas un wheel CPU-natif. "
                    "vLLM ne pourra pas résoudre sa plateforme "
                    "(current_platform) et échouera avec 'Device "
                    "string must not be empty'. Vérifiez "
                    "requirements.txt : la ligne doit être "
                    "'vllm==<version>+cpu' (pas seulement "
                    "'vllm==<version>'), et que pip installe bien "
                    "depuis wheels.vllm.ai/<version>/cpu plutôt que "
                    "depuis PyPI."
                )

        try:
            from vllm import AsyncEngineArgs
            from vllm import AsyncLLMEngine
        except ImportError as exc:
            raise RuntimeError(
                "vLLM n'est pas installé ou "
                "indisponible dans cet environnement. "
                "Vérifiez requirements.txt et la "
                "présence d'un GPU compatible, ou du "
                "backend CPU natif de vLLM."
            ) from exc

        _assert_inference_backend_available()

        engine_kwargs = dict(
            model=_MERGED_MODEL_NAME,
            revision=_MERGED_MODEL_REVISION,
            tokenizer_revision=_MERGED_MODEL_REVISION,
            dtype="bfloat16",
            max_model_len=(
                runtime_config.max_input_tokens + runtime_config.max_output_tokens
            ),
            trust_remote_code=True,
        )

        if is_cpu_backend:
            # AsyncEngineArgs n'expose plus de paramètre "device" en
            # vLLM 0.25.1 (cf. traceback de prod : "AsyncEngineArgs.
            # __init__() got an unexpected keyword argument 'device'").
            # Rien à transmettre ici via engine_kwargs : la sélection
            # de plateforme CPU est assurée par le wheel installé
            # lui-même (suffixe de version "+cpu", vérifié plus haut),
            # pas par un kwarg ici. Le dimensionnement du KV cache
            # reste piloté par
            # VLLM_CPU_KVCACHE_SPACE (cf. Dockerfile), pas par un
            # paramètre d'AsyncEngineArgs — de même pour
            # gpu_memory_utilization, qui n'a pas de sens sur CPU et
            # n'est donc pas renseigné dans cette branche.
            pass
        else:
            engine_kwargs["gpu_memory_utilization"] = 0.85

        engine_args = AsyncEngineArgs(**engine_kwargs)

        logger.info(
            "Initializing vLLM AsyncLLMEngine "
            "(backend=%s, merged_model=%s, no dynamic LoRA)",
            "cpu" if is_cpu_backend else "cuda",
            _MERGED_MODEL_NAME,
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
    Génère une réponse via vLLM (AsyncLLMEngine), en servant
    directement le modèle fusionné (_MERGED_MODEL_NAME).

    Signature volontairement alignée sur
    generate.generate_response() (mêmes paramètres
    de sampling) afin de permettre un dispatch
    transparent depuis TriageEngine, sans changer
    l'API publique existante.
    """

    from vllm import SamplingParams
    import uuid

    engine = get_vllm_engine()
    tokenizer = get_vllm_tokenizer()

    prompt = build_chat_prompt_with_tokenizer(
        tokenizer=tokenizer,
        system_prompt=system_prompt,
        user_prompt=user_prompt,
    )

    sampling_params = SamplingParams(
        max_tokens=max_new_tokens,
        temperature=temperature,
        top_p=top_p,
        repetition_penalty=repetition_penalty,
    )

    resolved_request_id = request_id or str(uuid.uuid4())

    final_output = None

    async for output in engine.generate(
        prompt=prompt,
        sampling_params=sampling_params,
        request_id=resolved_request_id,
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
