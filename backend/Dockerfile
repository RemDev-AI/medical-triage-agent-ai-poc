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
    patchelf \
    && rm -rf /var/lib/apt/lists/*

# ==========================================================
# PYTHON DEPENDENCIES
# ==========================================================

COPY requirements.txt ./requirements.txt
RUN pip install --upgrade pip
RUN pip install --no-cache-dir -r requirements.txt

# --- Garde-fou : échouer le BUILD (pas seulement le runtime) si vLLM
# ne résout pas en plateforme CPU. Corrige le problème rencontré en
# prod où un wheel "vanilla" (sans CPU) avait été résolu silencieusement
# par pip, provoquant un crash tardif au premier appel /triage/
# (AttributeError: init_cpu_memory_env).
#
# NB (mise à jour) : on NE VÉRIFIE PLUS que '+cpu' apparaisse dans
# vllm.__version__. Constaté en build réel : le wheel vllm-0.25.1+cpu
# s'installe correctement (torch/torchaudio/torchvision résolus en
# +cpu, nom de fichier wheel confirmé +cpu dans les logs pip), mais
# vllm.__version__ rapporte "0.25.1" SANS le suffixe — ce suffixe
# n'est présent que dans le nom du fichier wheel, pas dans la
# métadonnée interne du package pour ce build. Se fier au suffixe
# de version était donc un faux critère (faux positif de build).
# Le seul test fiable est la résolution réelle de la plateforme.
RUN python -c "\
import vllm; \
print('vLLM version installee :', vllm.__version__); \
from vllm.platforms import current_platform; \
print('Plateforme resolue :', current_platform); \
assert current_platform.is_cpu(), f'current_platform ne resout PAS en CPU (resolu: {current_platform!r}) — wheel vllm non-CPU probablement installe'"

# --- CORRECTIF : "cannot enable executable stack as shared object
# requires" (observé en prod sur _C_AVX512.abi3.so, et responsable en
# cascade de l'AttributeError 'init_cpu_memory_env' — cet op n'est
# jamais enregistré dans torch.ops._C car le .so qui le porte échoue
# à se charger sur l'environnement conteneurisé restreint de HF
# Spaces, qui refuse d'accorder une pile exécutable (mprotect
# PROT_EXEC) même si vllm/cpu.py catch cet ImportError et continue
# en dégradé). L'image de base python:3.12-slim tourne sur Debian
# trixie (13), dont la glibc >= 2.41 a rendu ce refus strict (avant,
# c'était juste toléré). On nettoie ici le flag ELF PT_GNU_STACK
# "exécutable" sur TOUS les .so de vllm (pas seulement _C_AVX512, par
# précaution) avec `patchelf --clear-execstack` — remplaçant moderne
# de `execstack -c`, qui a été RETIRÉ du dépôt Debian trixie (constaté
# en build : "E: Unable to locate package execstack"). patchelf reste
# disponible et packagé nativement sur trixie. Ceci ne modifie pas le
# comportement du code, juste une métadonnée ELF indiquant au noyau
# que le .so n'a en réalité pas besoin d'une pile exécutable — ce qui
# est vrai pour du code CPU généré par un compilateur standard (le
# flag est souvent un résidu de build,
# cf. https://github.com/vllm-project/vllm/issues/33675 et la
# discussion glibc >= 2.41 sur ce type d'erreur).
RUN find /usr/local/lib/python3.12/site-packages/vllm -name "*.abi3.so" -exec patchelf --clear-execstack {} \; \
    && find /usr/local/lib/python3.12/site-packages/vllm -name "*.abi3.so" -exec patchelf --print-execstack {} \;

# Garde-fou n°2 (diagnostic, NON bloquant) : le check précédent
# ("hasattr(torch.ops._C, 'init_cpu_memory_env')" juste après
# "import vllm") a été retiré car il donnait un FAUX POSITIF négatif
# — constaté en build réel. Raison : cet opérateur n'est enregistré
# dans torch.ops._C qu'après le chargement effectif de
# vllm._C_AVX512, qui n'a lieu qu'à l'INITIALISATION RÉELLE du worker
# CPU (vllm/v1/worker/cpu_worker.py), exécutée dans un sous-processus
# multiproc au démarrage du moteur — pas au moment d'un simple
# "import vllm" dans le process principal du build. Le test échouait
# donc systématiquement, qu'il y ait ou non un vrai problème,
# rendant le garde-fou inutilisable comme condition de succès/échec.
# On se contente ici d'un diagnostic informatif : tenter d'importer
# vllm._C_AVX512 directement pour vérifier, AU MOINS pour ce sous-
# module précis dans CE process, que le correctif patchelf a bien
# résolu l'erreur "cannot enable executable stack" — sans bloquer le
# build si ça ne prouve rien de plus (le test définitif reste
# l'exécution réelle du moteur au runtime).
RUN python -c "\
try: \
    import vllm._C_AVX512; \
    print('OK (diagnostic) : vllm._C_AVX512 importe sans erreur apres patchelf --clear-execstack'); \
except ImportError as e: \
    print(f'ATTENTION (diagnostic, non bloquant) : vllm._C_AVX512 echoue encore a l import : {e}')" || true

# ==========================================================
# APPLICATION
# ==========================================================
COPY . ./

# ==========================================================
# HUGGING FACE SPACES
# ==========================================================

ENV PORT=7860

# --- Réglages vLLM CPU backend ---

# NB : VLLM_TARGET_DEVICE n'est PAS lue par vLLM au runtime — c'est un
# réglage de COMPILATION du wheel (cf. vllm/setup.py). La conserver ici
# est inoffensif et cohérent avec la doc officielle vLLM CPU, mais elle
# n'a aucun effet sur la résolution de current_platform (celle-ci est
# déjà vérifiée au build ci-dessus, cf. garde-fou current_platform.is_cpu()).
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
