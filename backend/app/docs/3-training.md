# Documentation du Pipeline d'Entraînement 🧠⚙️

## 1. Introduction

Ce document décrit l'ensemble du pipeline d'entraînement du projet **Medical AI Triage Agent**.

L'objectif est de produire un modèle spécialisé dans le triage médical capable :
- d'évaluer un niveau d'urgence ;
- de générer une justification clinique ;
- de proposer des recommandations adaptées ;
- de limiter les hallucinations ;
- d'améliorer la sécurité des réponses.

Le pipeline suit les standards modernes de :
- LLMOps ;
- AI Engineering ;
- MLOps ;
- Fine-Tuning supervisé ;
- Alignement par préférences.

---

## 2. Architecture du Pipeline

```
Datasets RAW
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
Fine-Tuning LoRA
      │
      ▼
Modèle SFT
      │
      ▼
Dataset DPO
      │
      ▼
Alignement DPO
      │
      ▼
Modèle Final
      │
      ▼
Inference Engine
```

---

## 3. Modèle de Base

**Modèle sélectionné** : Qwen3-1.7B-Base

Motivations :
- excellent rapport qualité / performance ;
- coût d'entraînement réduit ;
- compatibilité LoRA et Hugging Face ;
- faible consommation GPU.

---

## 4. Datasets Utilisés

### 4.1 MediQA
Utilisé pour raisonnement médical et réponses cliniques.

### 4.2 FrenchMedMCQA
Connaissances médicales en français et validation clinique.

### 4.3 MedQuAD
Questions/réponses médicales et enrichissement du vocabulaire médical.

### 4.4 UltraMedical Preference Dataset
Alignement DPO et apprentissage des préférences.

---

## 5. Préparation des Données

### Standardisation
Tous les datasets sont convertis en JSONL UTF-8 avec un format unifié :
```json
{
  "instruction": "...",
  "input": "...",
  "output": "...",
  "confidence": 0.95
}
```

### Nettoyage
Suppression :
- doublons ;
- données corrompues ;
- réponses incomplètes ;
- données non médicales.

### Contrôle qualité
Validation :
- cohérence clinique ;
- cohérence linguistique ;
- qualité des annotations.

---

## 6. Anonymisation

- Détection PII avec Presidio ;
- Suppression des données personnelles ;
- Remplacement des identifiants sensibles.

Entités détectées :
- noms, emails, adresses, téléphones, identifiants médicaux.

---

## 7. Dataset SFT

**Objectif** : apprentissage supervisé du comportement attendu.

**Structure** :
```json
{
  "instruction": "Patient présente une douleur thoracique.",
  "output": "Urgence élevée ..."
}
```

**Taille** : ≈ 5000 exemples

**Splits** :
| Split      | Pourcentage |
|------------|-------------|
| Train      | 80 %        |
| Validation | 10 %        |
| Test       | 10 %        |

---

## 8. Fine-Tuning LoRA

**Objectif** : réduire le coût d'entraînement tout en conservant les performances.

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

## 9. Entraînement SFT

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
- MLflow ;
- Weights & Biases ;
- métriques : training loss, validation loss, learning rate, temps d'entraînement.

**Checkpoints** :
- Sauvegarde toutes les 500 étapes ;
- Reprise automatique, rollback et comparaison de modèles.

---

## 10. Dataset DPO

**Objectif** : aligner le modèle avec des préférences médicales sûres.

**Structure** :
```json
{
  "prompt": "...",
  "chosen": "...",
  "rejected": "..."
}
```

Exemple :
- Chosen : "Consultez immédiatement un service d'urgence."
- Rejected : "Attendez plusieurs jours avant consultation."

---

## 11. Entraînement DPO

Objectifs :
- Réduire hallucinations, réponses dangereuses, recommandations inadaptées ;
- Améliorer précision clinique, prudence médicale, cohérence des réponses.

**Paramètres** :
```yaml
beta: 0.1
learning_rate: 5e-5
epochs: 2
```

---

## 12. Évaluation

### Automatique
Mesures : Loss, Accuracy, Perplexity.

### Clinique
- Contrôle manuel du niveau d'urgence ;
- Qualité des recommandations ;
- Cohérence des réponses ;
- Sécurité clinique.

---

## 13. Optimisation GPU

- Quantization 4-bit, 8-bit ;
- bfloat16 (CUDA compatible GPU).

---

## 14. Gestion des Artefacts

Artefacts générés : checkpoints/, models/, logs/, metrics/, reports/

Versionnement : GitHub, Hugging Face Hub, MLflow

---

## 15. Publication Hugging Face

### Modèle
`medical-triage-agent-ai-poc-model` contenant : poids LoRA, tokenizer, configuration.

### Dataset
`medical-triage-agent-ai-poc-dataset` contenant : dataset SFT, dataset DPO, métadonnées.

---

## 16. Limites du POC

- Ne remplace pas un médecin ;
- Ne fournit pas de diagnostic médical ;
- Non certifié dispositif médical ;
- Ne doit pas être utilisé pour décision clinique réelle.

---

## 17. Roadmap Future

- Qwen3-4B, Qwen3-8B ;
- RAG médical ;
- Évaluation clinique automatisée ;
- RLHF ;
- Monitoring qualité temps réel.

---

## 18. Conclusion

Pipeline d'entraînement complet et reproductible : datasets spécialisés, anonymisation RGPD, Fine-Tuning LoRA, alignement DPO, monitoring complet.

Permet un modèle de triage médical robuste, traçable et aligné aux bonnes pratiques d'AI Engineering.
