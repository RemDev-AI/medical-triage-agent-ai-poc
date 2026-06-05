# Documentation du Pipeline d'Entraînement 🧠⚙️

## 1. Introduction

Ce document décrit l'ensemble du pipeline d'entraînement du projet **Medical AI Triage Agent**.

L'objectif est de produire un modèle spécialisé dans le triage médical capable :

- d'évaluer un niveau d'urgence ;
- de générer une justification clinique ;
- de proposer des recommandations adaptées ;
- de limiter les hallucinations ;
- d'améliorer la sécurité des réponses ;
- de garantir la traçabilité et la reproductibilité du pipeline.

Le projet suit une architecture moderne combinant :

- Hugging Face Hub ;
- Modal AI Infrastructure pour GPU global ;
- LLMOps ;
- MLOps ;
- AI Engineering ;
- Fine-Tuning supervisé ;
- Alignement par préférences.

---

## 2. Architecture Générale du Pipeline

```text
Hugging Face Datasets
        │
        ▼
Anonymisation RGPD
        │
        ▼
Préprocessing
        │
        ▼
Dataset SFT
        │
        ▼
Modal GPU Training
        │
        ▼
LoRA Fine-Tuning
        │
        ▼
Modèle SFT
        │
        ▼
Dataset DPO
        │
        ▼
Modal GPU DPO Training
        │
        ▼
Modèle Final
        │
        ▼
Hugging Face Models
        │
        ▼
Inference Engine
        │
        ▼
Hugging Face Space API
        │
        ▼
Hugging Face Space UI
```

---

## 3. Architecture Hugging Face + Modal

### 3.1 Hugging Face

Repositories principaux :

- **Models** : `medical-triage-agent-ai-poc-models`  
  Contient : modèle final, adaptateurs LoRA, tokenizer, configuration d'inférence, model card.

- **Datasets** : `medical-triage-agent-ai-poc-datasets`  
  Contient : datasets RAW, SFT, DPO, métadonnées, dataset card.

- **API** : `medical-triage-agent-ai-poc-api`  
  Docker Space hébergeant FastAPI, Inference Engine, Monitoring.

- **UI** : `medical-triage-agent-ai-poc-ui`  
  Streamlit Space hébergeant interface utilisateur, dashboard et historique.

### 3.2 Modal AI Infrastructure

Modal est utilisé pour tous les besoins GPU :

- Entraînement SFT et DPO ;
- Évaluation et benchmarks ;
- Batch inference ;
- Génération de datasets.

Avantages :

- GPU à la demande (A100/H100) ;
- Facturation à l’usage ;
- Scalabilité automatique ;
- Intégration Python native ;
- Sécurité par isolation et secrets vault.

---

## 4. Modèle de Base

**Modèle sélectionné** : `Qwen3-1.7B-Base`  

Motivations :

- Excellent rapport performance/coût ;
- Compatible PEFT et LoRA ;
- Faible consommation GPU ;
- Performances multilingues adaptées au français médical.

---

## 5. Datasets Utilisés

- **MediQA** : raisonnement clinique, réponses spécialisées.  
- **FrenchMedMCQA** : connaissances médicales francophones, validation clinique.  
- **MedQuAD** : enrichissement vocabulaire médical, génération de réponses.  
- **UltraMedical Preference Dataset** : alignement DPO, apprentissage préférences, sécurité clinique.

---

## 6. Préparation des Données

### Standardisation

Tous les datasets sont convertis en **JSONL UTF-8** avec un format unifié :

```json
{
  "instruction": "...",
  "input": "...",
  "output": "...",
  "confidence": 0.95
}
```

### Nettoyage

- Suppression des doublons ;
- Suppression des exemples incomplets ou corrompus ;
- Filtrage des données non médicales.

### Contrôle Qualité

- Cohérence clinique et linguistique ;
- Conformité RGPD ;
- Qualité des annotations.

---

## 7. Anonymisation RGPD

Technologies :

- Microsoft Presidio ;
- SpaCy FR (`fr_core_news_md`).

Entités détectées : noms, emails, adresses, téléphones, identifiants médicaux.  
Stratégies : mask, redact, replace.

---

## 8. Dataset SFT

**Objectif** : apprentissage supervisé du comportement attendu.  

**Taille** : ≈ 5 000 exemples.  

**Répartition** :

| Split      | Pourcentage |
|------------|-------------|
| Train      | 80 %        |
| Validation | 10 %        |
| Test       | 10 %        |

---

## 9. Fine-Tuning LoRA

**Objectif** : réduire les coûts GPU tout en conservant les performances.  

**Paramètres** :

```yaml
r: 16
lora_alpha: 32
lora_dropout: 0.05
bias: none
task_type: CAUSAL_LM
```

**Modules ciblés** : q_proj, k_proj, v_proj, o_proj, gate_proj, up_proj, down_proj

---

## 10. Entraînement SFT sur Modal GPU

**GPU principaux** : A100 80GB, H100.  

**Hyperparamètres** :

```yaml
learning_rate: 2e-4
batch_size: 4
gradient_accumulation_steps: 8
num_epochs: 3
warmup_ratio: 0.03
weight_decay: 0.01
```

**Monitoring** :

- MLflow, Weights & Biases, Modal Metrics ;
- Training/validation loss, learning rate, temps d'entraînement, consommation GPU, VRAM.

**Checkpoints** :

- Sauvegarde toutes les 500 steps ;
- Reprise automatique, rollback, comparaison d'expériences.

---

## 11. Dataset DPO

Alignement sur comportements médicaux sûrs.

**Structure** :

```json
{
  "prompt": "...",
  "chosen": "...",
  "rejected": "..."
}
```

---

## 12. Entraînement DPO sur Modal GPU

**Objectifs** :

- Réduction hallucinations et recommandations dangereuses ;
- Amélioration précision, prudence et sécurité clinique.

**Paramètres** :

```yaml
beta: 0.1
learning_rate: 5e-5
epochs: 2
```

---

## 13. Évaluation

**Automatique** : Loss, Accuracy, Perplexity, Latence.  
**Clinique** : contrôle niveau urgence, cohérence médicale, sécurité recommandations, stabilité réponses.

---

## 14. Optimisation GPU

- 4-bit et 8-bit Quantization ;
- bfloat16 ;
- Gradient Checkpointing ;
- LoRA.

Objectifs : réduction coûts, VRAM, accélération entraînement.

---

## 15. Gestion des Artefacts

- checkpoints/, models/, logs/, metrics/, reports/  
- Versionnement : GitHub, Hugging Face Hub, MLflow

---

## 16. Publication Hugging Face

- **Modèles** : `medical-triage-agent-ai-poc-models` (LoRA, tokenizer, configuration, model card)  
- **Datasets** : `medical-triage-agent-ai-poc-datasets` (SFT, DPO, métadonnées, dataset card)

---

## 17. Coûts Prévisionnels Modal

- Développement ponctuel : 5 à 20 USD  
- Entraînement complet POC : 20 à 100 USD selon nombre d’expériences, taille des datasets et GPU utilisé

---

## 18. Limites du POC

- Ne remplace pas un médecin ;
- Pas de diagnostic médical ;
- Non certifié dispositif médical ;
- Non destiné à usage clinique réel.

---

## 19. Roadmap Future

**Court terme** : amélioration datasets, augmentation corpus SFT, enrichissement DPO  
**Moyen terme** : RAG médical, évaluation clinique automatisée, optimisation Modal GPU  
**Long terme** : Qwen3-4B/8B, RLHF, Kubernetes, multi-GPU distribué

---

## 20. Conclusion

Pipeline d’entraînement robuste, traçable et scalable combinant :

- Hugging Face Hub ;
- Modal AI GPU Infrastructure ;
- LoRA, SFT, DPO ;
- MLflow, Weights & Biases.

Architecture optimisée pour coût, performance et standards MLOps modernes.
