# Rapport Final du Projet Medical AI Triage Agent 🏥🤖

## 1. Résumé Exécutif

Le projet **Medical AI Triage Agent** a été conçu comme un Proof of Concept (POC) visant à démontrer la faisabilité technique d'un système de triage médical assisté par intelligence artificielle.

L'architecture finale combine :

- un modèle de langage spécialisé ;
- un pipeline MLOps complet ;
- une API FastAPI sécurisée ;
- une interface utilisateur Streamlit ;
- une infrastructure GPU cloud via Modal ;
- une observabilité complète ;
- une documentation technique exhaustive.

L'objectif principal est de démontrer qu'une architecture moderne basée sur Hugging Face et Modal AI peut permettre la construction d'un système de triage médical robuste, reproductible et scalable.

---

## 2. Objectifs du Projet

### Objectifs Fonctionnels

- analyse de symptômes médicaux ;
- classification des urgences ;
- génération de recommandations ;
- justification des décisions du modèle ;
- suivi des performances ;
- amélioration de la sécurité des réponses.

### Objectifs Techniques

- intégration d'un LLM spécialisé ;
- Fine-Tuning LoRA ;
- alignement DPO ;
- automatisation CI/CD ;
- observabilité complète ;
- conformité RGPD ;
- optimisation GPU ;
- déploiement cloud moderne.

---

## 3. Architecture Générale

### Vue d'Ensemble

```text
Utilisateur
      │
      ▼
Hugging Face Space UI
      │
      ▼
Hugging Face Space API
      │
      ▼
Inference Engine
      │
      ├────────► Hugging Face Models
      │
      └────────► Modal GPU Infrastructure
                    │
                    ▼
              Qwen3 + LoRA
```

---

### Couche Présentation

Frontend Streamlit :

- saisie des symptômes ;
- affichage des résultats ;
- visualisation des métriques ;
- consultation de l'historique.

Repository :

```text
medical-triage-agent-ai-poc-ui
```

---

### Couche API

Backend FastAPI :

- validation ;
- authentification ;
- journalisation ;
- monitoring ;
- orchestration des modèles.

Repository :

```text
medical-triage-agent-ai-poc-api
```

---

### Couche IA

Inference Engine :

- chargement dynamique des modèles ;
- exécution GPU ;
- génération des réponses ;
- calcul du niveau d'urgence.

Technologies :

- Transformers ;
- PEFT ;
- vLLM ;
- Modal AI.

---

### Couche Données

Datasets utilisés :

- MediQA ;
- MedQuAD ;
- FrenchMedMCQA ;
- UltraMedical Preference Dataset.

Repository :

```text
medical-triage-agent-ai-poc-datasets
```

---

### Couche Modèles

Repository :

```text
medical-triage-agent-ai-poc-models
```

Contient :

- modèle Qwen3 ;
- adaptateurs LoRA ;
- tokenizer ;
- model card ;
- configurations.

---

### Couche Observabilité

- métriques ;
- monitoring ;
- alerting ;
- audit ;
- suivi GPU.

---

## 4. Réalisations Techniques

### Data Engineering

Mise en œuvre :

- ingestion datasets ;
- standardisation JSONL ;
- anonymisation RGPD ;
- preprocessing SFT ;
- preprocessing DPO ;
- validation qualité.

---

### Machine Learning

Mise en œuvre :

- configuration LoRA ;
- entraînement SFT ;
- alignement DPO ;
- tracking MLflow ;
- checkpoints ;
- optimisation GPU.

---

### Backend

Fonctionnalités :

- API REST ;
- validation Pydantic ;
- authentification JWT ;
- logging centralisé ;
- endpoints de monitoring ;
- audit.

---

### Frontend

Fonctionnalités :

- interface Streamlit ;
- dashboard ;
- historique ;
- visualisation métriques.

---

### Déploiement

Infrastructure :

- Docker ;
- GitHub Actions ;
- Hugging Face Spaces ;
- Hugging Face Hub ;
- Modal GPU Infrastructure.

---

## 5. Architecture Hugging Face + Modal

### Hugging Face

Le projet repose sur quatre repositories principaux :

#### API

```text
medical-triage-agent-ai-poc-api
```

#### UI

```text
medical-triage-agent-ai-poc-ui
```

#### Models

```text
medical-triage-agent-ai-poc-models
```

#### Datasets

```text
medical-triage-agent-ai-poc-datasets
```

---

### Modal AI

Modal est utilisé pour :

- entraînement LoRA ;
- entraînement DPO ;
- batch inference ;
- benchmarking ;
- optimisation GPU.

GPU principaux :

```text
NVIDIA A100 80GB
NVIDIA H100
```

---

## 6. Pipeline MLOps

```text
RAW DATA
    │
    ▼
ANONYMISATION
    │
    ▼
PREPROCESSING
    │
    ▼
DATASET SFT
    │
    ▼
MODAL GPU TRAINING
    │
    ▼
MODELE SFT
    │
    ▼
DATASET DPO
    │
    ▼
MODAL GPU DPO
    │
    ▼
MODELE FINAL
    │
    ▼
HUGGING FACE MODELS
    │
    ▼
DEPLOYMENT
    │
    ▼
MONITORING
```

Cette architecture garantit :

- reproductibilité ;
- traçabilité ;
- auditabilité ;
- industrialisation ;
- optimisation des coûts.

---

## 7. Sécurité et Conformité

### RGPD

Mesures mises en œuvre :

- anonymisation Presidio ;
- suppression des PII ;
- limitation de conservation ;
- journalisation contrôlée.

---

### Sécurité API

Mesures appliquées :

- JWT ;
- HTTPS ;
- CORS ;
- Rate Limiting ;
- validation stricte ;
- audit logging.

---

### Sécurité Infrastructure

Mesures :

- secrets GitHub ;
- secrets Hugging Face ;
- secrets Modal ;
- isolation des workloads GPU.

---

### Auditabilité

Traçabilité :

- logs ;
- métriques ;
- événements ;
- historiques d'entraînement ;
- déploiements.

---

## 8. Résultats Obtenus

### Fonctionnalités Disponibles

Le POC permet :

- triage médical simulé ;
- génération de réponses ;
- monitoring ;
- audit ;
- suivi des performances.

---

### Infrastructure

Infrastructure opérationnelle :

- CI/CD automatisé ;
- déploiement cloud ;
- monitoring ;
- alerting ;
- gestion GPU.

---

### Pipeline IA

Pipeline complet :

- SFT ;
- DPO ;
- LoRA ;
- évaluation ;
- publication automatique.

---

## 9. Analyse des Coûts

### Coût d'Infrastructure

Architecture retenue :

```text
Hugging Face + Modal
```

---

### Développement

Estimation :

```text
5 à 20 USD
```

---

### Entraînement Complet

Estimation :

```text
20 à 100 USD
```

---

### Optimisations Réalisées

- LoRA ;
- Quantization ;
- bfloat16 ;
- A100 à la demande ;
- H100 uniquement pour besoins avancés.

---

## 10. Limites Identifiées

Le projet reste un POC.

Limites :

- absence de certification médicale ;
- absence de certification HDS ;
- absence de validation clinique réelle ;
- couverture fonctionnelle limitée ;
- absence de supervision médicale réglementaire.

---

## 11. Perspectives d'Évolution

### Court Terme

- amélioration du dataset ;
- augmentation du corpus SFT ;
- enrichissement DPO ;
- amélioration de l'évaluation.

---

### Moyen Terme

- intégration RAG médical ;
- optimisation GPU ;
- amélioration du monitoring ;
- évaluation automatisée.

---

### Long Terme

- Qwen3-4B ;
- Qwen3-8B ;
- RLHF ;
- Kubernetes ;
- multi-GPU distribué ;
- certification réglementaire.

---

## 12. Bilan du Projet

### Points Forts

- architecture modulaire ;
- pipeline MLOps complet ;
- documentation exhaustive ;
- reproductibilité ;
- observabilité intégrée ;
- optimisation GPU ;
- coûts maîtrisés.

---

### Points d'Amélioration

- enrichissement des données ;
- validation clinique ;
- montée en charge ;
- certification réglementaire ;
- amélioration des benchmarks.

---

## 13. Livrables Produits

### Documentation

```text
1-architecture.md
2-rgpd.md
3-training.md
4-deployment.md
5-api.md
6-modal.md
7-final_report.md
```

---

### Code Source

```text
Backend FastAPI
Frontend Streamlit
Inference Engine
Pipeline ML
Scripts de déploiement
Monitoring
Tests
```

---

### Infrastructure

```text
Docker
GitHub Actions
Hugging Face Spaces
Hugging Face Models
Hugging Face Datasets
Modal AI Infrastructure
```

---

## 14. Technologies Utilisées

### Intelligence Artificielle

- Qwen3-1.7B ;
- Transformers ;
- PEFT ;
- LoRA ;
- DPO ;
- vLLM.

---

### Backend

- FastAPI ;
- Pydantic ;
- Uvicorn.

---

### Frontend

- Streamlit ;
- Plotly.

---

### MLOps

- GitHub Actions ;
- MLflow ;
- Weights & Biases.

---

### Infrastructure

- Docker ;
- Hugging Face ;
- Modal AI.

---

## 15. Conclusion Générale

Le projet **Medical AI Triage Agent** démontre la faisabilité d'une plateforme moderne de triage médical basée sur les grands modèles de langage.

L'architecture retenue associe :

- Hugging Face Hub ;
- Hugging Face Spaces ;
- Modal AI Global GPU Infrastructure ;
- FastAPI ;
- Streamlit ;
- LoRA ;
- DPO.

L'ensemble du projet applique les bonnes pratiques :

- AI Engineering ;
- LLMOps ;
- MLOps ;
- DevSecOps ;
- Documentation-as-Code.

Le résultat obtenu constitue une base solide pour une future industrialisation, sous réserve des validations réglementaires, cliniques et de sécurité nécessaires dans le domaine médical.
