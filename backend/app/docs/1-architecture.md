# Architecture Technique du POC Agent de Triage Médical 🏥🤖

## 1. Introduction
Ce document décrit l’architecture complète du projet **Medical AI Triage Agent**. Il suit les standards professionnels de :

- **LLMOps & AI Engineering**
- **MLOps**
- **Déploiement cloud IA**
- **CI/CD industriel**

L’objectif est de garantir une traçabilité, une sécurité clinique et un pipeline stable pour l’agent de triage.

---

## 2. Pipeline de Développement

Le développement suit un pipeline incrémental et validé à chaque étape :

1. **Infrastructure** (Docker, CI/CD, Git)
2. **Data Engineering** (datasets RAW, anonymisation, preprocessing SFT/DPO)
3. **Training LLM** (LoRA, SFT, DPO, tracking)
4. **Inference Engine** (chargement modèles, moteur de triage)
5. **API FastAPI** (routes, middlewares sécurité)
6. **Frontend Streamlit** (UI, pages métier)
7. **Déploiement Hugging Face** (models & Spaces)
8. **Monitoring & Observabilité** (GPU, API, alerting)
9. **Tests** (unitaires, intégration, sécurité)
10. **Documentation finale** (architecture, RGPD, training, API, déploiement, rapport final)

---

## 3. Infrastructure & CI/CD

- **Gestion Git**
  - `.gitignore`, branches GitFlow, conventions commits
- **Docker**
  - Backend et frontend containers
- **CI/CD via GitHub Actions**
  - Lint
  - Tests unitaires et d’intégration
  - Build Docker et déploiement automatisé
- **Stack Python**
  - PyTorch, Transformers, PEFT, vLLM
  - FastAPI backend
  - Streamlit frontend

---

## 4. Data Engineering

### 4.1 Datasets RAW 🌐
- **Sources** : MediQA, FrenchMedMCQA, MedQuAD, UltraMedical Preference Dataset
- **Standardisation** : JSONL, HF Datasets, UTF-8, champs unifiés

### 4.2 Anonymisation RGPD 🔐
- **Outils** : Microsoft Presidio, SpaCy FR (`fr_core_news_md`)
- **Cibles** : noms, emails, adresses, identifiants médicaux
- **Stratégies** : mask, redact, replace

### 4.3 Preprocessing SFT/DPO 🧠
- **SFT Dataset** : 5000 paires instruction/réponse avec métadonnées et niveaux de confiance
- **DPO Dataset** : `chosen/rejected` et scores qualité clinique
- **Splits** : train, validation, test, clinical evaluation

---

## 5. Training LLM 🧠⚙️

### 5.1 LoRA Configuration 🔧
- Paramètres : `rank`, `alpha`, `dropout`, `target_modules`
- Intégration PEFT pour fine-tuning

### 5.2 SFT Training 🚀
- Modèle de base : `Qwen3-1.7B-Base`
- Tracking : MLflow et Weights & Biases
- Checkpoints, early stopping et restauration

### 5.3 DPO Training 🧬
- Alignement préférentiel : `chosen/rejected`
- Optimisation clinique et contrôles sécurité (hallucinations, recommandations dangereuses)

---

## 6. Inference Engine ⚡

### 6.1 Loaders modèles 🧠
- `model_loader.py`, `tokenizer_loader.py`, `quantization_loader.py`
- Optimisations GPU : 4-bit, 8-bit, bfloat16

### 6.2 Moteur de triage 🏥
- Prompts : symptômes, antécédents, priorités urgence
- Génération : priorité, justification, recommandations

---

## 7. API FastAPI 🌐

### 7.1 Routes
- `/health`, `/triage`, `/generate`, `/audit`
- Validation JSON avec Pydantic
- Schemas request/response

### 7.2 Middleware sécurité 🔒
- JWT, rate limiting, CORS
- Logging auditables (requêtes, timestamps, sessions)

---

## 8. Frontend Streamlit 🎨

### 8.1 Base UI
- Layout global : sidebar, navigation, branding médical
- Connexion backend FastAPI

### 8.2 Pages métier 🩺
- `1_Home.py`, `2_Triage.py`, `3_History.py`, `4_Metrics.py`, `5_Admin.py`
- Fonctionnalités : formulaire patient, génération réponses, dashboard latence/logs

---

## 9. Déploiement Hugging Face ☁️

- **Backend (API)** : FastAPI, Docker, vLLM
- **Frontend (UI)** : Streamlit
- URL Spaces : `medical-triage-agent-ai-poc-api`, `medical-triage-agent-ai-poc-ui`
- Validation : accessibilité, inférence fonctionnelle, latence acceptable

---

## 10. Monitoring & Observabilité 📊

- GPU : VRAM, CUDA, throughput
- API : temps réponse, erreurs, trafic
- Alerting : alertes automatiques sur anomalies

---

## 11. Tests 🧪

- Unitaires : loaders, API, anonymisation
- Intégration : pipeline complet, inference, frontend/backend
- Sécurité : injections, accès, endpoints
- Validation humaine obligatoire pour couverture complète

---

## 12. Résultats attendus

- **Backend** : pipeline IA complet, LoRA, DPO, sécurité clinique, API scalable
- **Frontend** : interface médicale Streamlit, dashboard temps réel, historique triage
- **Infrastructure** : Docker, CI/CD, Hugging Face, monitoring
- **Gouvernance** : RGPD, auditabilité, reproductibilité, versioning complet

---

## 13. Conclusion

Cette architecture suit les standards MLOps modernes et les workflows Hugging Face. L’approche par validation humaine garantit :

- Sécurisation des livrables
- Qualité du code
- Limitation des erreurs critiques
- Traçabilité complète du projet
