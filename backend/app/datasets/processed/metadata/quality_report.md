# Quality Report — Medical Datasets

## Vérifications effectuées

### SFT
- suppression doublons ;
- contrôle champs vides ;
- normalisation UTF-8 ;
- validation schéma.

### DPO
- génération chosen/rejected ;
- scoring qualité clinique ;
- scoring sécurité.

### Splits
- train ;
- validation ;
- test ;
- clinical evaluation.

## Contrôles qualité médicale

- détection réponses dangereuses ;
- limitation hallucinations ;
- alignement clinique.

## Recommandations

Validation humaine médicale obligatoire avant :
- fine-tuning ;
- production ;
- déploiement clinique.
