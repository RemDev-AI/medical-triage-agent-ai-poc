# Modal AI Infrastructure Documentation 🚀

## 1. Introduction

Ce document décrit l'intégration de **Modal AI Infrastructure** dans le projet **Medical AI Triage Agent**.

Modal constitue la couche d'exécution GPU du projet et complète l'écosystème Hugging Face utilisé pour :

- l'entraînement des modèles ;
- l'alignement DPO ;
- les évaluations ;
- les benchmarks ;
- l'inférence GPU à la demande.

L'objectif est de disposer d'une infrastructure :

- scalable ;
- reproductible ;
- sécurisée ;
- optimisée en coûts ;
- adaptée aux standards MLOps modernes.

---

## 2. Architecture Générale

### Vue d'Ensemble

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
                                    │
                                    ▼
                    ┌────────────────────────────────┐
                    │ Hugging Face Space API         │
                    │ medical-triage-agent-ai-poc-api│
                    └───────────────┬────────────────┘
                                    │
                                    ▼
                         ┌────────────────────┐
                         │ Inference Engine   │
                         └─────────┬──────────┘
                                   │
                ┌──────────────────┴─────────────────┐
                ▼                                    ▼

    ┌──────────────────────┐          ┌──────────────────────┐
    │ Hugging Face Models  │          │ Modal GPU Cluster    │
    │ LoRA + Tokenizer     │          │ A100 / H100          │
    └──────────────────────┘          └──────────────────────┘

                ▲
                │

    ┌──────────────────────┐
    │ Hugging Face Dataset │
    └──────────────────────┘
```

---

## 3. Rôle de Modal dans le Projet

Modal est utilisé exclusivement pour les traitements nécessitant des ressources GPU.

### Cas d'Usage

#### Entraînement SFT

```text
Qwen3 + LoRA
```

Objectifs :

- Fine-Tuning supervisé ;
- réduction des coûts ;
- optimisation GPU.

---

#### Entraînement DPO

Objectifs :

- alignement des réponses ;
- amélioration de la sécurité ;
- réduction des hallucinations.

---

#### Batch Inference

Utilisation :

- évaluations ;
- benchmarks ;
- génération de datasets.

---

#### Inférence Temps Réel

Utilisation :

- génération des réponses médicales ;
- exécution du moteur de triage ;
- calcul du niveau d'urgence.

---

## 4. Architecture Technique Modal

### Services Utilisés

Modal fournit :

- exécution serverless GPU ;
- gestion des conteneurs ;
- autoscaling ;
- gestion des secrets ;
- monitoring intégré.

---

### Architecture Applicative

```text
Modal App
    │
    ├── Training Service
    │
    ├── DPO Service
    │
    ├── Evaluation Service
    │
    └── Inference Service
```

---

## 5. GPU Sélectionnés

### GPU Principal

```text
NVIDIA A100 80GB
```

Utilisation :

- Fine-Tuning LoRA ;
- DPO ;
- validation.

Avantages :

- excellent rapport coût/performance ;
- large disponibilité ;
- mémoire importante.

---

### GPU Avancé

```text
NVIDIA H100
```

Utilisation :

- benchmarks ;
- expérimentations ;
- montée en charge.

Avantages :

- performances maximales ;
- génération rapide ;
- entraînement accéléré.

---

## 6. Stratégie de Sélection GPU

### Développement

GPU recommandé :

```text
A100
```

---

### Validation

GPU recommandé :

```text
A100
```

---

### Benchmarking

GPU recommandé :

```text
H100
```

---

### Production

GPU recommandé :

```text
A100
```

H100 utilisé uniquement lorsque le gain de performance justifie le coût supplémentaire.

---

## 7. Déploiement Modal

### Installation

```bash
pip install modal
```

---

### Authentification

```bash
modal setup
```

---

### Variables d'Environnement

```env
MODAL_TOKEN_ID=xxxxxxxx

MODAL_TOKEN_SECRET=xxxxxxxx
```

---

### Vérification

```bash
modal profile current
```

---

## 8. Déploiement des Services

### Entraînement

```bash
modal deploy backend/app/deployment/modal/modal_training.py
```

---

### DPO

```bash
modal deploy backend/app/deployment/modal/modal_dpo.py
```

---

### Inference

```bash
modal deploy backend/app/deployment/modal/modal_inference.py
```

---

### Monitoring

```bash
modal app list
```

---

## 9. Gestion des Secrets

### Secrets Requis

```text
HF_TOKEN

WANDB_API_KEY

MLFLOW_TRACKING_URI

MODAL_TOKEN_ID

MODAL_TOKEN_SECRET
```

---

### Bonnes Pratiques

- ne jamais stocker les secrets dans Git ;
- utiliser Modal Secrets ;
- utiliser GitHub Secrets ;
- rotation périodique.

---

## 10. Monitoring

### Métriques GPU

Suivi :

- utilisation GPU ;
- mémoire GPU ;
- temps d'exécution ;
- temps d'inférence.

---

### Métriques Modèle

Suivi :

- latence ;
- tokens/seconde ;
- temps de génération ;
- taux d'erreur.

---

### Métriques Infrastructure

Suivi :

- nombre d'instances ;
- temps CPU ;
- mémoire ;
- autoscaling.

---

## 11. Alerting

### Critique

- GPU indisponible ;
- erreur d'inférence ;
- échec d'entraînement ;
- dépassement de quota.

---

### Warning

- forte consommation GPU ;
- latence élevée ;
- saturation mémoire ;
- coût inhabituel.

---

## 12. Sécurité

### Isolation

Modal exécute chaque workload dans un environnement isolé.

Avantages :

- sécurité renforcée ;
- séparation des exécutions ;
- réduction des risques.

---

### Authentification

Mécanismes :

- Tokens Modal ;
- gestion des rôles ;
- contrôle d'accès.

---

### Chiffrement

Protection :

- données en transit ;
- secrets ;
- connexions API.

---

## 13. Optimisation des Coûts

### Stratégies Utilisées

#### LoRA

Réduction importante du coût d'entraînement.

---

#### Quantization

Utilisation :

```text
4-bit
8-bit
```

---

#### bfloat16

Réduction de la mémoire GPU utilisée.

---

#### Autoscaling

Arrêt automatique des ressources inutilisées.

---

## 14. Estimation des Coûts

### Développement

```text
5 à 20 USD
```

---

### Fine-Tuning SFT

```text
10 à 50 USD
```

---

### DPO

```text
5 à 30 USD
```

---

### POC Complet

```text
20 à 100 USD
```

Les coûts dépendent :

- du nombre d'expériences ;
- du volume des données ;
- du GPU sélectionné ;
- du temps d'exécution.

---

## 15. Intégration avec Hugging Face

### Models

Repository :

```text
medical-triage-agent-ai-poc-models
```

Modal charge automatiquement :

- modèle ;
- tokenizer ;
- adaptateurs LoRA.

---

### Datasets

Repository :

```text
medical-triage-agent-ai-poc-datasets
```

Modal consomme :

- datasets SFT ;
- datasets DPO ;
- métadonnées.

---

### Spaces

Repositories :

```text
medical-triage-agent-ai-poc-api

medical-triage-agent-ai-poc-ui
```

Modal fournit :

- les ressources GPU ;
- les services d'inférence ;
- les traitements lourds.

---

## 16. Reprise Après Incident

### Sauvegardes

Éléments sauvegardés :

- modèles ;
- checkpoints ;
- métriques ;
- logs ;
- configurations.

---

### Procédure

1. restauration du modèle ;
2. restauration des checkpoints ;
3. redéploiement Modal ;
4. validation des services ;
5. validation du monitoring.

---

## 17. Bonnes Pratiques MLOps

### Développement

- utiliser des environnements reproductibles ;
- versionner les modèles ;
- documenter les expériences.

---

### Production

- monitorer les GPU ;
- monitorer les coûts ;
- surveiller la latence ;
- conserver les logs.

---

### Sécurité

- rotation des secrets ;
- moindre privilège ;
- audit régulier.

---

## 18. Limites du POC

Cette infrastructure :

- n'est pas certifiée HDS ;
- n'est pas certifiée dispositif médical ;
- n'est pas destinée à une utilisation clinique réelle ;
- nécessite une supervision humaine.

---

## 19. Roadmap Future

### Court Terme

- optimisation des jobs Modal ;
- amélioration monitoring.

---

### Moyen Terme

- multi-GPU ;
- optimisation coûts.

---

### Long Terme

- orchestration hybride Modal + Kubernetes ;
- modèles plus volumineux ;
- RLHF ;
- monitoring avancé.

---

## 20. Conclusion

Modal AI constitue la couche GPU officielle du projet Medical AI Triage Agent.

L'association :

- Hugging Face Models ;
- Hugging Face Datasets ;
- Hugging Face Spaces ;
- Modal AI Infrastructure ;

permet de construire une architecture moderne, scalable, sécurisée et optimisée en coûts.

Cette approche respecte les bonnes pratiques d'AI Engineering, LLMOps, MLOps et DevSecOps tout en conservant une infrastructure légère adaptée à un Proof of Concept.
