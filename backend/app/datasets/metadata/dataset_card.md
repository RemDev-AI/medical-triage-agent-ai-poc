# Dataset Card — POC Agent IA Triage Médical

## Description

Dataset médical unifié destiné au fine-tuning :
- SFT ;
- DPO ;
- triage médical.

## Sources

- MediQA
- FrenchMedMCQA
- MedQuAD
- UltraMedical

## Objectif

Créer un assistant médical :
- robuste ;
- aligné ;
- sécurisé ;
- conforme RGPD.

## Format standard

```json
{
  "id": "...",
  "instruction": "...",
  "response": "...",
  "source": "...",
  "language": "...",
  "metadata": {}
}
