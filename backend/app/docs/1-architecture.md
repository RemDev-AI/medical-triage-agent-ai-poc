# Architecture Technique du POC Agent de Triage Médical 🏥🤖

## 1. Introduction
Ce document décrit l’architecture complète du projet **Medical AI Triage Agent**.

L’architecture inclut désormais :
- **Hugging Face Spaces & Hub** pour API et UI ;
- **Modal AI Infrastructure** pour exécution GPU scalable globale ;
- Traçabilité et sécurité clinique intégrées.

---

## 2. Pipeline de Développement

1. **Infrastructure**
   - Docker, CI/CD, Git
   - **Modal AI GPU Infrastructure** pour entraînement et inférence
2. **Data Engineering**
3. **Training LLM**
4. **Inference Engine**
5. **API FastAPI**
6. **Frontend Streamlit**
7. **Déploiement Hugging Face**
8. **Monitoring & Observabilité**
9. **Tests**
10. **Documentation finale**

---

## 3. Infrastructure & Modal GPU

- **Modal AI Global GPU**
  - GPU choisis selon charge et type modèle (ex: A100, H100)
  - Scalabilité globale : execution sur clusters distribués
  - Monitoring intégré
  - Sécurité : isolation containers, secrets vault

- **CI/CD via GitHub Actions**
  - Lint
  - Tests unitaires et d’intégration
  - Build Docker
  - Déploiement Hugging Face + Modal GPU

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
- Entraînement LoRA/SFT/DPO via **Modal GPU** pour performance optimisée
- 
---

## 6. Inference Engine ⚡
- Loaders modèles et moteur triage exécutés sur **Modal GPU** si latence critique
- Optimisations GPU : 4-bit, 8-bit, bfloat16

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

## 9. Déploiement Hugging Face ☁️
- Backend : Hugging Face Space API + Docker
- Frontend : Hugging Face Space UI
- Secrets et tokens sécurisés via Modal AI secrets vault

## 10. Monitoring & Observabilité 📊
- GPU usage monitoring via Modal AI
- API metrics via FastAPI
- Alerting automatisé

## 11. Tests 🧪

- Unitaires : loaders, API, anonymisation
- Intégration : pipeline complet, inference, frontend/backend
- Sécurité : injections, accès, endpoints
- Validation humaine obligatoire pour couverture complète

## 12. Résultats attendus
- Backend : pipeline IA complet sur Modal GPU, LoRA, DPO
- Frontend : Streamlit
- Infrastructure : Docker, CI/CD, Hugging Face, Modal GPU
- Gouvernance : RGPD, auditabilité, versioning complet

## 13. Conclusion
Architecture robuste combinant Hugging Face et Modal AI pour un POC de triage médical scalable et sécurisé.
