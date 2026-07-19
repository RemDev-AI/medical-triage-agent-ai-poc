# ==========================================================
# medical-triage-agent-ai-poc/backend/Dockerfile
# Hugging Face Space API Container — OPTION B
# vLLM CPU backend natif (x86), pour comparaison avec
# l'option A (Dockerfile, transformers CPU via model_loader.py)
# ==========================================================

# ATTENTION — À LIRE AVANT UTILISATION :
#
# 1. Le backend CPU de vLLM recommande AVX512 (AVX2 accepté
#    depuis les roues pré-compilées >= 0.17.0, avec
#    performances réduites). Sur l'infra partagée gratuite
#    d'un Hugging Face Space "CPU basic", le jeu d'instructions
#    du CPU alloué n'est PAS garanti et PEUT NE PAS supporter
#    AVX512. Vérifier avec la commande de diagnostic fournie
#    en bas de ce fichier avant de considérer cette option
#    comme fiable.
#
# 2. Un tier "CPU basic" (2 vCPU) est très en dessous du
#    profil visé par ce backend (pensé pour du parallélisme
#    OpenMP multi-cœurs). Latence probablement dégradée par
#    rapport à l'option A (transformers), à mesurer.
#
# 3. Cette variante installe une version PINÉE (pas un commit
#    flottant type vllm-ci-test-repo, qui est réservé au
#    benchmarking interne de l'équipe vLLM, cf. discussion
#    précédente) — reproductibilité garantie.
#
# 4. Build avec la RACINE du dépôt comme contexte, cf.
#    Dockerfile.hf pour l'explication complète.
# ==========================================================

FROM python:3.12-slim

# ==========================================================
# ENVIRONMENT
# ==========================================================

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

ENV APP_HOME=/app

# Hugging Face / Modal
ENV HF_HOME=/tmp/huggingface

WORKDIR ${APP_HOME}

# ==========================================================
# SYSTEM DEPENDENCIES
# ==========================================================
#
# gcc/g++ >= 12.3.0 recommandé par la doc vLLM pour le
# backend CPU (build de certains kernels). libnuma-dev requis
# pour le binding CPU/mémoire (VLLM_CPU_OMP_THREADS_BIND).
# python3-dev requis pour la compilation de certaines
# extensions natives.

RUN apt-get update && apt-get install -y \
    build-essential \
    gcc \
    g++ \
    curl \
    git \
    libnuma-dev \
    python3-dev \
    numactl \
    && rm -rf /var/lib/apt/lists/*

# ==========================================================
# PYTHON DEPENDENCIES
# ==========================================================

COPY requirements.txt ./requirements.txt
RUN pip install --upgrade pip
RUN pip install --no-cache-dir -r requirements.txt

# ==========================================================
# APPLICATION
# ==========================================================
COPY . ./

# ==========================================================
# HUGGING FACE SPACES
# ==========================================================

ENV PORT=7860

# --- Réglages vLLM CPU backend ---

# CORRECTIF (mise à jour) : contrairement à une hypothèse
# précédente, VLLM_TARGET_DEVICE n'est PAS lue par vLLM au runtime
# — c'est un réglage de COMPILATION du wheel (cf. vllm/setup.py).
# La conserver ici est inoffensif et reste cohérent avec la doc
# officielle vLLM CPU, mais elle ne suffit PAS à faire résoudre
# current_platform en CpuPlatform : cette résolution dépend
# uniquement de la présence du suffixe "+cpu" dans la chaîne de
# version du wheel vLLM réellement installé (cf. requirements.txt :
# "vllm==0.25.1+cpu"). Le vrai correctif à l'erreur "RuntimeError:
# Device string must not be empty" observée en prod est là, pas ici.
ENV VLLM_TARGET_DEVICE=cpu

# Réserve de la mémoire pour le KV cache (en Go). À ajuster
# selon la RAM réellement disponible sur le tier du Space
# (16 Go annoncés ; laisser de la marge pour l'OS/FastAPI/
# spacy/presidio également chargés en mémoire).
ENV VLLM_CPU_KVCACHE_SPACE=4

# Sur 2 vCPU, un seul thread OpenMP a du sens (pas de gain à
# binder plusieurs threads sur un tier aussi contraint) ;
# à ajuster si le tier réel expose plus de cœurs.
ENV VLLM_CPU_OMP_THREADS_BIND=0

# dtype recommandé sur CPU (float16 CPU est instable selon
# la doc vLLM) — cohérent avec bfloat16 déjà utilisé dans
# vllm_engine.py (AsyncEngineArgs(dtype="bfloat16", ...)).

EXPOSE 7860

# ==========================================================
# HEALTHCHECK
# ==========================================================
#
# start-period augmenté à 60s : le premier appel de
# génération déclenche le chargement paresseux du moteur
# vLLM (cf. get_vllm_engine() lazy init) ; sur CPU, le
# chargement initial du modèle + adaptateur peut être plus
# long que sur GPU. /health reste néanmoins un endpoint léger
# qui ne charge pas le modèle (cf. main.py), donc ce délai
# est une marge de sécurité, pas une dépendance stricte.

HEALTHCHECK --interval=30s \
            --timeout=10s \
            --start-period=60s \
            --retries=5 \
            CMD curl -f http://localhost:${PORT}/health || exit 1

# ==========================================================
# STARTUP
# ==========================================================

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "7860"]

# ==========================================================
# DIAGNOSTIC — à exécuter MANUELLEMENT avant d'adopter cette
# option, pour vérifier le support AVX512/AVX2 du CPU alloué
# par le Space (ne fait pas partie du build) :
#
#   docker exec -it <container_id> bash -c "cat /proc/cpuinfo | grep -o 'avx512[a-z0-9_]*' | sort -u"
#   docker exec -it <container_id> bash -c "cat /proc/cpuinfo | grep -o 'avx2' | head -1"
#
# Si aucun flag avx512* n'apparaît, seules les roues AVX2
# (>= 0.17.0) seront utilisables, avec des performances
# réduites par rapport à AVX512.
# ==========================================================
