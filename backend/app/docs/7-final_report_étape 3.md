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

- L'adaptateur LoRA final (checkpoint DPO, `RemDev-AI/medical-triage-agent-ai-poc-models/checkpoints/dpo/checkpoint-dpo-32`) est chargé **directement par vLLM** via `--enable-lora`, sans fusion préalable (`merge_and_unload`), conformément au format déjà produit à l'étape 2. *(Correction de ce rapport : le sous-dossier `ref/` mentionné précédemment ici correspond au modèle de référence figé utilisé uniquement pour le calcul de la divergence KL pendant l'entraînement DPO — ce n'est jamais un artefact de production ; voir la docstring de `vllm_engine.py`.)*
- `generate.py` dispatche automatiquement vers vLLM ou vers le chemin `transformers` historique selon `runtime_config.use_vllm`, sans modification de l'API publique de `TriageEngine`.
- En environnement CI (sans GPU), `USE_VLLM=false` est forcé pour les tests automatisés, qui mockent `TriageEngine` et `generate_response()` au niveau des dépendances FastAPI (`get_triage_engine`, `get_generation_context`) — voir section 8.

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
- ~~Signature `InferenceClient.triage(symptoms: str, ...)` vs `TriageRequest.symptoms: List[str]`~~ : point résolu par la migration vers l'inférence locale (section 8). `TriageRequest.symptoms`/`medical_history` restent des `str` (schéma API inchangé), convertis en listes à un seul élément dans `routes/triage.py` avant l'appel à `TriageEngine.run_triage()`.
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

## 8. Addendum — Migration `InferenceClient` vers l'inférence locale

Un audit ultérieur (post-rédaction du présent rapport) a confirmé que `InferenceClient` (client HTTP vers un backend d'inférence externe) demeurait, jusqu'à cette migration, le seul point d'entrée réellement utilisé par les routes `/triage` et `/generate` — alors même que le moteur d'inférence embarqué (`generate.py`, `triage_engine.py`, `vllm_engine.py`) était déjà fonctionnel. La migration a été réalisée dans l'ordre suivant :

1. **Routes** — `routes/triage.py` appelle désormais directement `TriageEngine.run_triage()` ; `routes/inference.py` appelle directement `generate_response()`. Le module `api/dependencies/inference.py` ne contient plus de client HTTP : il expose deux dépendances FastAPI paresseuses (chargement au premier appel réel, et non au démarrage de l'application) : `get_triage_engine()` et `get_generation_context()`.
2. **Chargement du modèle** — Le modèle Transformers + adaptateur LoRA ne sont chargés qu'une seule fois (singleton thread-safe), lors du premier appel effectif à l'une de ces dépendances, jamais dans le `lifespan` de `app/main.py` — afin de ne jamais bloquer le démarrage de l'API, les health checks, ni l'exécution des tests, suivant le même principe que `vllm_engine.get_vllm_engine()`.
3. **Tests** — Les doubles de test (`FakeInferenceClient` et variantes) ont été remplacés par des doubles pour `TriageEngine` (surchargés via `app.dependency_overrides[get_triage_engine]`) et pour `generate_response()` (patché directement dans le module `routes/inference.py` via `monkeypatch`, celui-ci étant appelé comme fonction libre et non injecté).
4. **CI/CD** — La variable d'environnement `INFERENCE_API_URL` a été retirée de tous les jobs de `backend-ci.yml` (elle n'est plus consommée par aucun code applicatif). Le secret GitHub Actions correspondant peut être supprimé après vérification qu'aucun autre workflow (déploiement, health-check externe) n'en dépend encore.
5. **Documentation** — Le présent rapport (sections 3 et 6) et les commentaires de `api/schemas.py` ont été mis à jour pour ne plus référencer `InferenceClient`.

**Point de vigilance résiduel** : `HF_API_TOKEN` (utilisé par l'ancien `InferenceClient` pour l'authentification Bearer vers le backend distant) n'a pas été retiré de `backend-ci.yml` par prudence — son usage ailleurs dans le code n'a pas été vérifié exhaustivement lors de cette migration. À confirmer avant suppression définitive.
