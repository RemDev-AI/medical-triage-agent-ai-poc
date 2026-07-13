# ==========================================================
# medical-triage-agent-ai-poc/scripts/deploy_to_hf_space.sh
# ==========================================================

#!/usr/bin/env bash
# Déploiement manuel (fallback) vers un Hugging Face Space.
# Usage: ./deploy_to_hf_space.sh <api|ui> [--dry-run]
#
# Prérequis :
#   - variable d'environnement HF_TOKEN_06 exportée (jamais commitée)
#   - huggingface_hub[cli] installé (pip install "huggingface_hub[cli]")

set -euo pipefail

TARGET="${1:-}"
DRY_RUN="${2:-}"

API_SPACE="RemDev-AI/medical-triage-agent-ai-poc-api"
UI_SPACE="RemDev-AI/medical-triage-agent-ai-poc-ui"

if [[ -z "${HF_TOKEN_06:-}" ]]; then
  echo "Erreur: la variable d'environnement HF_TOKEN_06 n'est pas définie." >&2
  echo "Exporte-la avant d'exécuter ce script : export HF_TOKEN_06=hf_xxx" >&2
  exit 1
fi

case "$TARGET" in
  api)
    SRC_DIR="backend"
    DOCKERFILE="backend/Dockerfile.hf"
    REPO_ID="$API_SPACE"
    ;;
  ui)
    SRC_DIR="frontend"
    DOCKERFILE="frontend/Dockerfile.hf"
    REPO_ID="$UI_SPACE"
    ;;
  *)
    echo "Usage: $0 <api|ui> [--dry-run]" >&2
    exit 1
    ;;
esac

STAGE_DIR="$(mktemp -d)"
trap 'rm -rf "$STAGE_DIR"' EXIT

echo "==> Préparation du payload pour ${REPO_ID}"
rsync -a --delete \
  --exclude '.git' \
  --exclude '__pycache__' \
  --exclude '*.pyc' \
  --exclude 'app/tests' \
  --exclude 'app/datasets/raw' \
  "${SRC_DIR}/" "${STAGE_DIR}/"
cp "${DOCKERFILE}" "${STAGE_DIR}/Dockerfile"

if [[ "$DRY_RUN" == "--dry-run" ]]; then
  echo "==> [dry-run] Contenu qui serait poussé vers ${REPO_ID} :"
  find "$STAGE_DIR" -maxdepth 2 -type f
  exit 0
fi

echo "==> Push vers Hugging Face Space: ${REPO_ID}"
python3 - <<PYEOF
import os
from huggingface_hub import HfApi

api = HfApi(token=os.environ["HF_TOKEN_06"])
api.upload_folder(
    folder_path="${STAGE_DIR}",
    repo_id="${REPO_ID}",
    repo_type="space",
    commit_message="Manual deploy via deploy_to_hf_space.sh",
)
api.restart_space(repo_id="${REPO_ID}")
PYEOF

echo "==> Déploiement terminé pour ${REPO_ID}"
