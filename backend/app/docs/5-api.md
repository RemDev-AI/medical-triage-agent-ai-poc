# Documentation API 🌐

## 1. Introduction

Cette documentation décrit l'API REST du projet **Medical AI Triage Agent**, intégrant désormais **Modal AI GPU Infrastructure** pour l’inférence haute performance.

L'API permet :

- exécution des évaluations de triage ;
- accès aux métriques système (incluant l’utilisation GPU Modal) ;
- consultation de l'état de santé du service ;
- audit des opérations ;
- intégration avec des applications externes.

Technologies :

- FastAPI ;
- Pydantic ;
- JWT Authentication ;
- OpenAPI 3.0 ;
- vLLM / Modal GPU pour l’inférence.

---

## 2. Architecture API

```text
Client
  │
  ▼
FastAPI
  │
  ├── Authentication
  ├── Validation
  ├── Rate Limiting
  ├── Logging
  └── Monitoring
  │
  ▼
Inference Engine
  │
  ▼
Qwen3 + LoRA
  │
  ▼
Modal GPU
```

---

## 3. Base URL

### Développement

```text
http://localhost:8000
```

### Production

```text
https://medical-triage-agent-ai-poc-api.hf.space
```

---

## 4. Authentification

### Type

```text
Bearer JWT
```

### Header

```http
Authorization: Bearer <token>
```

### Exemple

```http
GET /api/v1/metrics
Authorization: Bearer eyJ...
```

---

## 5. Endpoints Disponibles

| Endpoint         | Méthode  | Description                  |
|------------------|----------|------------------------------|
| /health          | GET      | État du service              |
| /api/v1/triage   | POST     | Triage médical               |
| /api/v1/generate | POST     | Génération libre             |
| /api/v1/metrics  | GET      | Monitoring, incl. GPU Modal  |
| /api/v1/audit    | GET      | Audit système                |

---

## 6. Endpoint Health

### Requête

```http
GET /health
```

### Réponse

```json
{
  "status": "healthy",
  "version": "1.0.0",
  "model_loaded": true,
  "modal_gpu_ready": true
}
```

### Codes

| Code   | Description          |
|--------|----------------------|
| 200    | Service disponible   |
| 503    | Service indisponible |

---

## 7. Endpoint Triage

### Requête

```http
POST /api/v1/triage
```

### Payload

```json
{
  "age": 45,
  "gender": "male",
  "symptoms": ["chest pain","shortness of breath"],
  "medical_history": ["hypertension"]
}
```

### Réponse

```json
{
  "priority": "HIGH",
  "confidence": 0.94,
  "justification": "Possible cardiac event",
  "recommendation": "Seek emergency care immediately",
  "gpu_usage": 45
}
```

### Codes

| Code   | Description       |
|--------|-------------------|
| 200    | Succès            |
| 400    | Données invalides |
| 401    | Non autorisé      |
| 429    | Trop de requêtes  |
| 500    | Erreur serveur    |

---

## 8. Endpoint Generate

### Requête

```http
POST /api/v1/generate
```

### Payload

```json
{
  "prompt": "Explain hypertension risks."
}
```

### Réponse

```json
{
  "response": "Hypertension increases the risk...",
  "gpu_usage": 48
}
```

---

## 9. Endpoint Metrics

### Requête

```http
GET /api/v1/metrics
```

### Réponse

```json
{
  "latency_ms": 245,
  "requests_total": 1024,
  "error_rate": 0.02,
  "gpu_usage": 56,
  "modal_gpu_utilization": 52
}
```

---

## 10. Endpoint Audit

### Requête

```http
GET /api/v1/audit
```

### Réponse

```json
{
  "events": [
    {
      "timestamp": "2026-01-01T12:00:00",
      "event": "triage_request"
    }
  ]
}
```

---

## 11. Modèles de Données

### TriageRequest

```json
{
  "age": 0,
  "gender": "string",
  "symptoms": [],
  "medical_history": []
}
```

### TriageResponse

```json
{
  "priority": "LOW|MEDIUM|HIGH|CRITICAL",
  "confidence": 0.0,
  "justification": "string",
  "recommendation": "string",
  "gpu_usage": 0
}
```

---

## 12. Validation des Données

Règles :

- âge compris entre 0 et 120 ;
- symptômes obligatoires ;
- listes non nulles ;
- types strictement validés.

Validation réalisée par :

```text
Pydantic
```

---

## 13. Gestion des Erreurs

### Format Standard

```json
{
  "error": true,
  "code": "INVALID_REQUEST",
  "message": "Age must be between 0 and 120"
}
```

---

## 14. Sécurité API

Mesures implémentées :

- JWT Authentication ;
- HTTPS ;
- CORS ;
- Rate Limiting ;
- Validation Pydantic ;
- Journalisation sécurisée ;
- secrets Modal GPU pour inférence.

---

## 15. Rate Limiting

Limite recommandée :

```text
100 requêtes / minute
```

par utilisateur authentifié.

---

## 16. Observabilité

Métriques collectées :

- temps de réponse ;
- taux d'erreur ;
- nombre de requêtes ;
- utilisation GPU Modal ;
- consommation mémoire.

---

## 17. Documentation Interactive

### Swagger UI

```text
/docs
```

### ReDoc

```text
/redoc
```

---

## 18. Versionnement API

Convention :

```text
/api/v1/
```

Évolutions futures :

```text
/api/v2/
/api/v3/
```

---

## 19. Bonnes Pratiques d'Intégration

- utiliser HTTPS ;
- gérer les erreurs 429 ;
- implémenter des timeouts ;
- journaliser les réponses ;
- surveiller les métriques GPU.

---

## 20. Conclusion

L'API Medical AI Triage Agent fournit :

- interface REST moderne ;
- validation robuste ;
- sécurité renforcée ;
- observabilité complète ;
- intégration simple avec des applications tierces ;
- support complet de l’inférence GPU via Modal AI.

Elle constitue la couche d'accès officielle aux fonctionnalités du moteur de triage médical et GPU.
