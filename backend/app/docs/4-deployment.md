# Guide de Déploiement 🚀

## 1. Introduction

Ce document décrit la stratégie complète de déploiement du projet **Medical AI Triage Agent**.

L'architecture repose sur une approche **Hugging Face + Modal AI Infrastructure** :

- Hugging Face Models ;
- Hugging Face Datasets ;
- Hugging Face Space API ;
- Hugging Face Space UI ;
- Modal GPU Infrastructure.

Objectifs :

- backend FastAPI scalable ;
- interface Streamlit accessible ;
- infrastructure reproductible ;
- pipeline CI/CD automatisé ;
- monitoring complet ;
- optimisation des coûts GPU.

---

## 2. Architecture de Déploiement

```text
                           ┌────────────────────┐
                           │     Utilisateur    │
                           └─────────┬──────────┘
                                     │
                                     ▼
                    ┌────────────────────────────────┐
                    │ Hugging Face Space UI          │
                    │ medical-triage-agent-ai-poc-ui │
                    └───────────────┬────────────────┘
                                    │ REST API
                                    ▼
                    ┌────────────────────────────────┐
                    │ Hugging Face Space API         │
                    │ medical-triage-agent-ai-poc-api│
                    └───────────────┬────────────────┘
                                    │
                                    ▼
                    ┌────────────────────────────────┐
                    │ Inference Engine               │
                    │ FastAPI + vLLM + LoRA          │
                    └───────────────┬────────────────┘
                                    │
                   ┌────────────────┴────────────────┐
                   ▼                                 ▼

      ┌───────────────────────┐        ┌─────────────────────────┐
      │ Hugging Face Models   │        │ Modal GPU Infrastructure│
      │ medical-triage-agent- │        │ A100 / H100             │
      │ ai-poc-models         │        │ Training & Inference    │
      └───────────────────────┘        └─────────────────────────┘

                   ▲
                   │
      ┌────────────────────────────┐
      │ Hugging Face Datasets      │
      │ medical-triage-agent-ai-   │
      │ poc-datasets               │
      └────────────────────────────┘
```

---

## 3. Stratégie de Déploiement

### Répartition des responsabilités

| Composant             | Plateforme             |
|-----------------------|------------------------|
| API                   | Hugging Face Space API |
| Interface utilisateur | Hugging Face Space UI  |
| Modèles               | Hugging Face Models    |
| Datasets              | Hugging Face Datasets  |
| GPU Training          | Modal                  |
| GPU Inference         | Modal                  |
| CI/CD                 | GitHub Actions         |

---

## 4. Composants Déployés

### Backend API

Repository :

```text
medical-triage-agent-ai-poc-api
```

Responsabilités :

- API REST ;
- authentification ;
- validation des requêtes ;
- orchestration des modèles ;
- monitoring.

Technologies :

- FastAPI ;
- Uvicorn ;
- Pydantic ;
- Transformers ;
- PEFT ;
- vLLM.

---

### Frontend UI

Repository :

```text
medical-triage-agent-ai-poc-ui
```

Responsabilités :

- saisie patient ;
- affichage résultats ;
- dashboard monitoring ;
- historique.

Technologies :

- Streamlit ;
- Requests ;
- Plotly.

---

### Modèle IA

Repository :

```text
medical-triage-agent-ai-poc-models
```

Contenu :

- Qwen3-1.7B ;
- LoRA adapters ;
- tokenizer ;
- configuration d'inférence.

---

### Datasets

Repository :

```text
medical-triage-agent-ai-poc-datasets
```

Contenu :

- datasets RAW ;
- datasets SFT ;
- datasets DPO ;
- métadonnées.

---

## 5. Infrastructure Modal AI

### Objectif

Modal est utilisé exclusivement pour les ressources GPU.

Cas d'utilisation :

- Fine-Tuning SFT ;
- Fine-Tuning DPO ;
- Batch Inference ;
- Évaluation ;
- Benchmarking.

---

### GPU Utilisés

#### GPU principal

```text
NVIDIA A100 80GB
```

Utilisation :

- SFT ;
- DPO ;
- évaluations.

#### GPU avancé

```text
NVIDIA H100
```

Utilisation :

- benchmarking ;
- optimisation ;
- montée en charge.

---

### Avantages

- GPU à la demande ;
- facturation à l'usage ;
- autoscaling ;
- haute disponibilité ;
- gestion simplifiée.

---

## 6. Déploiement Local

### Prérequis

```text
Python >= 3.11
Docker >= 24
Git >= 2.40
```

---

### Installation

```bash
git clone https://github.com/<organization>/medical-triage-agent-ai-poc.git

cd medical-triage-agent-ai-poc

pip install -r requirements.txt
```

---

## 7. Déploiement Docker

### Backend

```bash
docker build -t medical-triage-api .

docker run \
-p 8000:8000 \
medical-triage-api
```

---

### Frontend

```bash
docker build -t medical-triage-ui .

docker run \
-p 8501:8501 \
medical-triage-ui
```

---

## 8. Variables d'Environnement

### Backend

```env
ENV=production

HF_TOKEN=xxxxxxxx

HF_MODEL_REPOSITORY=medical-triage-agent-ai-poc-models

HF_DATASET_REPOSITORY=medical-triage-agent-ai-poc-datasets

JWT_SECRET_KEY=xxxxxxxx

LOG_LEVEL=INFO

MODAL_TOKEN_ID=xxxxxxxx

MODAL_TOKEN_SECRET=xxxxxxxx
```

---

### Frontend

```env
API_BASE_URL=https://medical-triage-agent-ai-poc-api.hf.space

REQUEST_TIMEOUT=30
```

---

## 9. Déploiement Hugging Face

### Space API

Repository :

```text
medical-triage-agent-ai-poc-api
```

Type :

```text
Docker Space
```

Responsabilités :

- FastAPI ;
- Inference Engine ;
- Monitoring.

---

### Space UI

Repository :

```text
medical-triage-agent-ai-poc-ui
```

Type :

```text
Streamlit Space
```

Responsabilités :

- interface utilisateur ;
- dashboards ;
- historique.

---

## 10. Déploiement des Modèles

Repository :

```text
medical-triage-agent-ai-poc-models
```

Contient :

- modèle final ;
- LoRA ;
- tokenizer ;
- configuration ;
- model card.

---

## 11. Déploiement des Datasets

Repository :

```text
medical-triage-agent-ai-poc-datasets
```

Contient :

- dataset SFT ;
- dataset DPO ;
- dataset card ;
- métadonnées.

---

## 12. Déploiement Modal

### Authentification

Secrets nécessaires :

```text
MODAL_TOKEN_ID

MODAL_TOKEN_SECRET
```

---

### Déploiement

Exemple :

```bash
modal deploy backend/app/deployment/modal/modal_inference.py
```

---

### Vérification

```bash
modal app list
```

---

### Logs

```bash
modal app logs
```

---

## 13. CI/CD GitHub Actions

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
Build
 │
 ▼
Publish Models
 │
 ▼
Deploy HF Spaces
 │
 ▼
Deploy Modal
 │
 ▼
Monitoring Validation
```

---

### Validation

Outils :

- Ruff ;
- Black ;
- MyPy.

---

### Tests

- unitaires ;
- intégration ;
- sécurité ;
- inférence.

---

### Déploiement

Publication automatique :

- Hugging Face Models ;
- Hugging Face Datasets ;
- Hugging Face Spaces ;
- Modal GPU.

---

## 14. Monitoring

### API

Mesures :

- temps de réponse ;
- trafic ;
- erreurs ;
- disponibilité.

---

### GPU

Mesures :

- VRAM ;
- utilisation CUDA ;
- température ;
- throughput.

---

### Modèle

Mesures :

- latence ;
- temps de génération ;
- nombre de requêtes ;
- tokens/seconde.

---

### Modal

Mesures :

- temps GPU ;
- coût d'exécution ;
- utilisation mémoire ;
- nombre d'instances.

---

## 15. Alerting

### Critique

- API indisponible ;
- modèle indisponible ;
- erreur GPU ;
- échec déploiement.

### Warning

- latence élevée ;
- saturation VRAM ;
- taux d'erreur élevé ;
- coût anormalement élevé.

---

## 16. Sécurité

Mesures appliquées :

- JWT ;
- HTTPS ;
- CORS ;
- Rate Limiting ;
- Validation Pydantic ;
- Logs d'audit ;
- Secrets Modal ;
- Secrets Hugging Face.

---

## 17. Reprise après Incident

### Sauvegardes

- modèles ;
- datasets ;
- logs ;
- métriques ;
- configurations.

---

### Procédure

1. restauration du modèle ;
2. restauration des datasets ;
3. restauration des secrets ;
4. redéploiement Hugging Face ;
5. redéploiement Modal ;
6. validation monitoring.

---

## 18. Scalabilité

### Backend

- FastAPI ;
- autoscaling ;
- multi-instances.

---

### Inference

- vLLM ;
- Tensor Parallelism ;
- Modal GPU Autoscaling.

---

### Infrastructure

- A100 ;
- H100 ;
- multi-GPU.

---

## 19. Validation du Déploiement

Checklist :

- API accessible ;
- UI accessible ;
- modèle chargé ;
- datasets accessibles ;
- Modal actif ;
- monitoring actif ;
- alertes fonctionnelles ;
- tests validés ;
- CI/CD opérationnel.

---

## 20. Coûts Prévisionnels

### Développement

```text
5 à 20 USD
```

---

### Entraînement complet POC

```text
20 à 100 USD
```

---

### Production POC

Dépend :

- trafic ;
- nombre d'inférences ;
- GPU utilisés ;
- durée d'exécution.

---

## 21. Limites du POC

Le déploiement actuel :

- n'est pas certifié HDS ;
- n'est pas certifié dispositif médical ;
- n'est pas destiné à un usage clinique réel ;
- nécessite une supervision humaine.

---

## 22. Conclusion

L'architecture de déploiement combine :

- Hugging Face Hub ;
- Hugging Face Spaces ;
- Modal AI Global GPU Infrastructure ;
- GitHub Actions ;
- FastAPI ;
- Streamlit ;
- vLLM.

Cette architecture fournit :

- reproductibilité ;
- observabilité ;
- sécurité ;
- automatisation ;
- optimisation des coûts GPU ;
- scalabilité moderne conforme aux standards MLOps et LLMOps.
