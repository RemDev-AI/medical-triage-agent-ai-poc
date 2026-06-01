# Guide de Déploiement 🚀

## 1. Introduction

Ce document décrit la stratégie complète de déploiement du projet **Medical AI Triage Agent**.

L'objectif est de fournir :

- un backend FastAPI scalable ;
- une interface Streamlit accessible ;
- une infrastructure reproductible ;
- un pipeline CI/CD automatisé ;
- une observabilité complète.

---

## 2. Architecture de Déploiement

```text
┌─────────────────────┐
│     Utilisateur     │
└──────────┬──────────┘
           │
           ▼
┌─────────────────────┐
│  Frontend Streamlit │
│   Hugging Face UI   │
└──────────┬──────────┘
           │ REST API
           ▼
┌─────────────────────┐
│   Backend FastAPI   │
│ Hugging Face Space  │
└──────────┬──────────┘
           │
           ▼
┌─────────────────────┐
│   Inference Engine  │
│     Qwen3 LoRA      │
└──────────┬──────────┘
           │
           ▼
┌─────────────────────┐
│ Hugging Face Hub    │
│ Models + Datasets   │
└─────────────────────┘
```

## 3. Composants Déployés

### Backend

Responsabilités :
- API REST
- authentification
- validation des requêtes
- chargement du modèle
- monitoring

Technologies :
- FastAPI
- Uvicorn
- Pydantic
- PyTorch
- Transformers
- PEFT

### Frontend

Responsabilités :
- saisie patient
- affichage résultats
- dashboard monitoring
- historique des requêtes

Technologies :
- Streamlit
- Requests
- Plotly

### Modèle IA

Composants :
- Qwen3-1.7B
- Adaptateurs LoRA
- Tokenizer
- Configuration d'inférence

---

## 4. Déploiement Local

### Prérequis

- Python >= 3.11
- Docker >= 24
- Git >= 2.4

### Installation

```bash
git clone https://github.com/<organization>/medical-triage-agent-ai-poc.git
cd medical-triage-agent-ai-poc
pip install -r requirements.txt
```

---

## 5. Déploiement Docker

### Backend

```bash
docker build -t triage-api .
docker run -p 8000:8000 triage-api
```

### Frontend

```bash
docker build -t triage-ui .
docker run -p 8501:8501 triage-ui
```

---

## 6. Variables d'Environnement

### Backend

```env
ENV=production
JWT_SECRET_KEY=xxxxxxxx
HF_TOKEN=xxxxxxxx
MODEL_ID=medical-triage-agent-ai-poc-model
LOG_LEVEL=INFO
```

### Frontend

```env
API_BASE_URL=https://triage-api-url
REQUEST_TIMEOUT=30
```

---

## 7. Déploiement Hugging Face

### Backend Space

Nom :

```text
medical-triage-agent-ai-poc-api
```

Type : Docker Space

### Frontend Space

Nom :

```text
medical-triage-agent-ai-poc-ui
```

Type : Streamlit Space

### Secrets Hugging Face

- HF_TOKEN
- JWT_SECRET_KEY
- API_BASE_URL

---

## 8. Déploiement du Modèle

Repository :

```text
medical-triage-agent-ai-poc-model
```

Contient :
- modèle de base
- poids LoRA
- tokenizer
- configuration

---

## 9. Déploiement des Datasets

Repository :

```text
medical-triage-agent-ai-poc-dataset
```

Contient :
- dataset SFT
- dataset DPO
- documentation
- métadonnées

---

## 10. CI/CD GitHub Actions

Pipeline :

```text
Push
 │
 ▼
Lint
 │
 ▼
Tests
 │
 ▼
Build Docker
 │
 ▼
Deploy
```

### Étapes

#### Validation
- Ruff
- Black
- MyPy

#### Tests
- Unitaires
- Intégration
- Sécurité

#### Build
- Backend Docker
- Frontend Docker

#### Déploiement
- Hugging Face Spaces
- Hugging Face Hub

---

## 11. Monitoring

### API
- temps de réponse
- trafic
- erreurs
- disponibilité

### GPU
- VRAM
- utilisation CUDA
- throughput

### Modèle
- latence
- temps génération
- nombre requêtes

---

## 12. Alerting

### Critique
- API indisponible
- modèle non chargé
- erreur GPU

### Warning
- latence élevée
- saturation mémoire
- taux erreur élevé

---

## 13. Sécurité

Mesures appliquées :
- JWT
- HTTPS
- CORS
- Rate Limiting
- Validation Pydantic
- Logs d'audit

---

## 14. Reprise après Incident

### Sauvegardes
- modèles
- datasets
- logs
- métriques

### Restauration
1. restauration modèle ;
2. restauration configuration ;
3. redémarrage services ;
4. validation monitoring.

---

## 15. Scalabilité

### Backend
- Load Balancer
- Kubernetes
- Autoscaling

### Inference
- vLLM
- Tensor Parallelism
- GPU dédiés

### Monitoring
- Prometheus
- Grafana
- AlertManager

---

## 16. Validation du Déploiement

Checklist :
- API accessible
- Frontend accessible
- Modèle chargé
- Monitoring actif
- Alertes fonctionnelles
- Tests validés

---

## 17. Limites du POC

Le déploiement actuel :
- n'est pas certifié HDS ;
- n'est pas certifié dispositif médical ;
- n'est pas destiné à une production clinique réelle.

---

## 18. Conclusion

L'architecture de déploiement du Medical AI Triage Agent fournit :
- reproductibilité ;
- observabilité ;
- sécurité ;
- automatisation ;
- scalabilité.

Elle constitue une base solide pour une industrialisation future selon les standards MLOps modernes.
