# Conformité RGPD du POC Agent de Triage Médical 🔐

## 1. Introduction

Ce document décrit les mesures de conformité mises en œuvre dans le projet **Medical AI Triage Agent** afin de respecter les exigences du Règlement Général sur la Protection des Données (RGPD - UE 2016/679).

L'objectif principal du projet est de développer un système d'assistance au triage médical basé sur l'intelligence artificielle tout en garantissant :
- la protection des données personnelles ;
- la confidentialité des informations médicales ;
- la minimisation des risques de réidentification ;
- l'auditabilité des traitements ;
- la transparence du système.

## 2. Cadre réglementaire applicable
Le projet s'appuie sur :
- RGPD (Règlement UE 2016/679)
- Principes de Privacy by Design
- Principes de Privacy by Default
- Recommandations CNIL
- Bonnes pratiques OWASP API Security
- Documentation Microsoft Presidio

## 3. Classification des données traitées

### 3.1 Données autorisées
- symptômes déclarés
- âge du patient
- sexe biologique
- antécédents médicaux
- traitements en cours
- niveau de douleur
- informations contextuelles médicales

### 3.2 Données interdites
- nom complet
- adresse postale
- numéro de téléphone
- adresse email
- numéro de sécurité sociale
- identifiant patient réel
- coordonnées GPS précises
- documents d'identité

## 4. Architecture de protection des données

Entrée utilisateur -> Détection PII (Presidio) -> Anonymisation -> Validation RGPD -> Pipeline IA -> Réponse de triage

## 5. Anonymisation des données

### 5.1 Technologies utilisées
- Microsoft Presidio Analyzer
- Microsoft Presidio Anonymizer
- SpaCy French Model
- Règles personnalisées médicales

### 5.2 Entités détectées
- PERSON, EMAIL_ADDRESS, PHONE_NUMBER, LOCATION, IP_ADDRESS, MEDICAL_ID, DATE_TIME

### 5.3 Stratégies d'anonymisation
- Mask, Redact, Replace

## 6. Politique de conservation des données
- Journaux applicatifs : 30 jours
- Logs d'audit : 90 jours
- Datasets d'entraînement : anonymisés, versionnés, stockés séparément

## 7. Sécurité des traitements
- Chiffrement TLS pour les données en transit
- Chiffrement disque pour les données stockées
- JWT Authentication et rotation des secrets
- Contrôle d'accès par rôle
- Protection API : CORS sécurisé, Rate Limiting, validation Pydantic, logs d'accès, protection contre injections

## 8. Journalisation et auditabilité
- Traces conservées : timestamp, session id, endpoint, durée traitement, statut
- Exclusion : PII et données médicales complètes

## 9. Privacy by Design
- Minimisation : seules les données nécessaires sont collectées
- Limitation de finalité : triage, amélioration du système, métriques techniques
- Limitation de conservation : suppression automatique après expiration
- Intégrité : protection contre altération et accès non autorisé

## 10. Gestion des incidents
- Étape 1 : Identification
- Étape 2 : Isolation
- Étape 3 : Analyse des logs d'audit
- Étape 4 : Évaluation de l'impact
- Étape 5 : Notification réglementaire

## 11. Limites du POC
- Absence de DPO dédié
- Absence de certification HDS
- Absence d'hébergement médical agréé
- Absence d'analyse d'impact RGPD complète

## 12. Responsabilités
- Équipe IA : datasets, anonymisation, sécurité modèle
- Équipe Backend : sécurité API, authentification, journalisation
- Équipe DevOps : infrastructure, secrets, monitoring, sauvegardes

## 13. Conclusion
Le POC Medical AI Triage Agent intègre des mécanismes de protection conformes au RGPD : minimisation, anonymisation, sécurité, auditabilité, transparence.
