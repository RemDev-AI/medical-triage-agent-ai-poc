# RGPD Anonymization Pipeline

Pipeline :
PII Detection
↓
Presidio Analyzer
↓
Anonymization
↓
Validation
↓
Audit Logs

## Technologies

- Microsoft Presidio
- SpaCy FR
- Regex médicales
- Audit logging

## Stratégies supportées

- mask
- redact
- replace

## Entités détectées

- PERSON
- EMAIL_ADDRESS
- PHONE_NUMBER
- LOCATION
- MEDICAL_RECORD_NUMBER
- PATIENT_ID
- FRENCH_SOCIAL_SECURITY

## Objectif

Garantir :
- conformité RGPD ;
- suppression PII ;
- auditabilité ;
- reproductibilité.
