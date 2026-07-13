# Pipeline CI/CD — medical-triage-agent-ai-poc

## Schéma du pipeline

```
push/PR sur backend/** ──► backend-ci.yml   (lint → tests unitaires → sécurité → intégration)
push/PR sur frontend/** ──► frontend-ci.yml  (lint → smoke test Streamlit)
                                    │
                    (branche main uniquement)
                                    ▼
                        docker-build-push.yml
              (attend le succès de backend-ci → build + scan Trivy
               → push GHCR pour backend et frontend)
                                    │
                                    ▼
                       deploy-huggingface.yml
        (gate go/no-go → push vers HF Space API → push vers
         HF Space UI → restart des Spaces → smoke test post-déploiement)

security-audit.yml : exécution hebdomadaire + sur chaque push/PR
  (gitleaks, pip-audit, scan Trivy des images buildées localement)
```

**Note :** aucun job de ce pipeline n'entraîne ni ne réentraîne le modèle.
L'entraînement (SFT/DPO) est réalisé en amont (étape 2) sur Colab (GPU T4 free),
avec tracking Weights & Biases, puis publié sur
`RemDev-AI/medical-triage-agent-ai-poc-models`. Ce pipeline consomme ces
checkpoints déjà validés.

## Secrets GitHub requis (Settings → Secrets and variables → Actions)

| Secret                 | Usage                                                                                         |
|------------------------|-----------------------------------------------------------------------------------------------|
| `HF_TOKEN_06`          | Token Hugging Face (write) pour push vers les Spaces/Models/Datasets                          |
| `WANDB_API_KEY`        | Weights & Biases — tracking/monitoring/logs/graphiques (training + événements de déploiement) |
| `GITLEAKS_LICENSE`     | Licence Gitleaks (optionnel, requis pour orgs)                                                |
| `GITHUB_TOKEN`         | Fourni automatiquement par GitHub Actions                                                     |

> ⚠️ Point de vigilance : ne jamais committer de token en clair. Tous les secrets
> ci-dessus doivent être stockés uniquement dans GitHub Secrets, jamais dans
> `.env` versionné, ni dans les logs de workflow (les steps utilisant `HF_TOKEN_06`
> et `WANDB_API_KEY` évitent tout `echo` de leur valeur).
>
> Note d'architecture : Modal n'est plus utilisé dans le pipeline. L'entraînement
> (étape 2) s'appuie sur le GPU T4 gratuit de Google Colab ; ce pipeline CI/CD
> (étape 3) ne fait que consommer les artefacts déjà publiés sur Hugging Face
> (`RemDev-AI/medical-triage-agent-ai-poc-models`), il ne relance aucun entraînement.

## Checklist go/no-go avant déploiement production

Le job `gate-check` dans `deploy-huggingface.yml` matérialise cette checklist :

- [x] `backend-ci.yml` : lint, tests unitaires, tests de sécurité/injections,
      tests d'intégration — tous verts
- [x] `docker-build-push.yml` : images backend/frontend construites et scannées
      (Trivy, seuil CRITICAL/HIGH bloquant)
- [ ] Revue manuelle des métriques de latence (voir étape tests de charge)
- [ ] Validation fonctionnelle de la Space UI en environnement pilote
- [ ] Documentation des limites d'usage à jour (`docs/2-rgpd.md`, `docs/5-api.md`)

Pour un déploiement en production stricte, il est recommandé de transformer
`deploy-huggingface.yml` en `workflow_dispatch` uniquement (retirer le
déclenchement automatique `workflow_run`) et d'ajouter un `environment:
production` avec règle d'approbation manuelle dans les paramètres du repo
GitHub (Settings → Environments).

## Déploiement manuel (fallback)

```bash
export HF_TOKEN_06=hf_xxx
./scripts/deploy_to_hf_space.sh api --dry-run   # vérifier le contenu
./scripts/deploy_to_hf_space.sh api             # déployer réellement
./scripts/deploy_to_hf_space.sh ui
```
