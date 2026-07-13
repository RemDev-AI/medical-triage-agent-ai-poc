# Rapport final — Étape 3 : Déploiement, tests et audits

## 1. Périmètre traité

Conformément au cahier des charges de l'étape 3 :

1. Tests de latence, de robustesse et audits de traçabilité des interactions.
2. Intégration vLLM (in-process, `AsyncLLMEngine`) pour l'inférence optimisée.
3. Correction des anomalies bloquantes identifiées durant l'analyse préalable, qui empêchaient un déploiement Docker/CI fonctionnel.

Les fichiers déjà validés à l'étape 3 (workflows CI/CD `.github/workflows/*` hors `backend-ci.yml`, `scripts/deploy_to_hf_space.sh`) n'ont pas été modifiés.

## 2. Anomalies corrigées

| # | Fichier(s) | Correction |
|---|---|---|
| 1 | `backend/Dockerfile`, `Dockerfile.hf` | `CMD` JSON multi-lignes invalide → forme exec sur une seule ligne |  
| 2 | `Dockerfile`, `Dockerfile.hf`, `docker-compose.yml` | Contexte de build aligné sur la racine du dépôt pour préserver les imports `backend.app.*` |  
| 3 | `backend/app/monitoring/gpu_monitor.py`| Ajout de `increment_request()`, absent alors qu'appelé par `TriageEngine.run_triage()` |
| 4 | `backend/app/api/routes/inference.py`, `triage.py` | Suppression du double comptage `request_tracker`/`latency_monitor` (déjà assuré par `AuditLoggingMiddleware`) |
| 5 | `backend/app/main.py` | Suppression de l'inclusion en double de `monitoring_router` |
| 6 | `backend/app/llm/inference/triage_engine.py` | `confidence_score` désormais calculé (auparavant absent) |

## 3. Intégration vLLM

Approche retenue : **`AsyncLLMEngine` en process** (`backend/app/llm/inference/vllm_engine.py`), activée via `runtime_config.use_vllm` (variable d'environnement `USE_VLLM`, déjà présente dans `hf_space_runtime.py`).

- L'adaptateur LoRA final (checkpoint DPO, `RemDev-AI/medical-triage-agent-ai-poc-models/ref`) est chargé **directement par vLLM** via `--enable-lora`, sans fusion préalable (`merge_and_unload`), conformément au format déjà produit à l'étape 2.
- `generate.py` dispatche automatiquement vers vLLM ou vers le chemin `transformers` historique selon `runtime_config.use_vllm`, sans modification de l'API publique de `TriageEngine`.
- En environnement CI (sans GPU), `USE_VLLM=false` est forcé pour les tests automatisés, qui mockent entièrement l'`InferenceClient` HTTP.

## 4. Traçabilité des interactions

Nouveau module `backend/app/monitoring/audit_store.py` : stockage persistant JSON Lines, alimenté par `AuditLoggingMiddleware` pour chaque requête HTTP. L'endpoint `GET /audit/` lit désormais ce stockage réel (pagination via `?limit=`), remplaçant les données mockées.

**Limite d'usage documentée (point 13)** : ce journal est local au conteneur et non répliqué. En environnement Hugging Face Space, il est réinitialisé à chaque redémarrage du Space. Il ne doit pas constituer l'unique source de vérité pour un audit réglementaire de long terme — une externalisation (base de données, stockage objet) est recommandée avant une mise en production définitive.

Ce mécanisme reste distinct du logger RGPD (`backend/app/anonymization/audit_logger.py`), dédié à la détection/anonymisation PII.

## 5. Tests ajoutés

Nouveau dossier `backend/app/tests/performance/`, intégré au pipeline CI (`backend-ci.yml`, job `performance-tests`, exécuté après `integration-tests`) :

- `test_latency.py` : mesure de latence par requête, latence sous charge (p95), cohérence avec `/monitoring/latency`.
- `test_robustness.py` : validation des entrées, comportement en cas de panne du backend d'inférence, non-régression sur les compteurs de trafic et sur `gpu_monitor.increment_request()`.
- `test_audit_trail.py` : persistance réelle des interactions, comportement de l'endpoint `/audit/`.

## 6. Points de vigilance non traités dans ce périmètre

- `dependencies.py` (`validate_priority`) reste non câblé dans les routes — aucune route ne l'utilise actuellement ; à statuer séparément si un contrôle de priorité applicatif est requis.
- Signature `InferenceClient.triage(symptoms: str, ...)` vs `TriageRequest.symptoms: List[str]` : incohérence de typage à clarifier côté service d'inférence externe (hors périmètre backend traité ici).
- Les secrets/clés d'accès (point 11) sont gérés via variables d'environnement (`.env`, secrets GitHub Actions) ; aucune clé en dur n'a été introduite.

## 7. Checklist go / no-go (point 10)

| Critère | Statut |
|---|---|
| Build Docker fonctionnel (CMD valide, imports résolus) | ✅ Corrigé |
| CI/CD : lint, tests unitaires, sécurité, intégration | ✅ Existant, validé |
| CI/CD : tests de latence/robustesse/traçabilité | ✅ Ajouté (`performance-tests`) |
| Intégration vLLM opérationnelle | ⚠️ Code prêt — **à valider sur environnement GPU réel** (non testable en CI) |
| Traçabilité des interactions | ✅ Stockage réel, limite d'usage documentée |
| Secrets protégés (point 11) | ✅ Variables d'environnement / secrets CI |
| Surveillance post-déploiement (point 12) | ⚠️ Alertes en mémoire (`alert_manager`) — non persistantes ; recommandé : export vers un système externe avant production |
| Documentation des limites d'usage (point 13) | ✅ Documentée (section 4) |

**Recommandation** : GO pour un déploiement en environnement pilote, sous réserve de la validation manuelle de l'inférence vLLM sur l'infrastructure GPU cible (Hugging Face Space), qui ne peut être vérifiée par la CI actuelle (absence de GPU sur les runners).
