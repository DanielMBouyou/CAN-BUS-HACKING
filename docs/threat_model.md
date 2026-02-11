# 🔒 Threat Model — CAN-Stealth-Attack-AI Lab

## 1. Périmètre

Ce modèle de menace couvre un réseau CAN automobile simulé composé de 4 ECUs virtuelles communiquant sur un bus CAN virtuel (`vcan0`).

## 2. Actifs

| Actif | Description | Criticité |
|-------|-------------|-----------|
| ECU Engine (0x100) | Contrôle RPM moteur | **Critique** |
| ECU ABS (0x110) | Contrôle vitesse roues | **Critique** |
| ECU Steering (0x120) | Contrôle direction | **Critique** |
| ECU Cluster (0x130) | Affichage tableau de bord | Élevé |
| Bus CAN (vcan0) | Medium de communication | **Critique** |

## 3. Menaces (STRIDE)

### 3.1 Spoofing (Usurpation)
- **Attaque** : Injection de frames CAN avec un ID arbitraire usurpé
- **Impact** : Prise de contrôle d'une fonction véhicule
- **Probabilité** : Élevée (pas d'authentification CAN native)

### 3.2 Tampering (Altération)
- **Attaque** : Modification des payloads en transit
- **Impact** : Valeurs capteurs corrompues
- **Probabilité** : Élevée

### 3.3 Repudiation (Répudiation)
- **Attaque** : Absence de traçabilité des frames
- **Impact** : Impossibilité d'identifier l'origine d'une attaque
- **Probabilité** : Élevée (CAN n'a pas de logging natif)

### 3.4 Information Disclosure
- **Attaque** : Écoute passive du bus CAN
- **Impact** : Extraction de données véhicule
- **Probabilité** : Élevée (bus partagé, pas de chiffrement)

### 3.5 Denial of Service
- **Attaque** : Saturation du bus par frames haute priorité
- **Impact** : ECUs légitimes bloquées
- **Probabilité** : Élevée

### 3.6 Elevation of Privilege
- **Attaque** : Accès physique → injection → contrôle fonctions critiques
- **Impact** : Contrôle total du véhicule
- **Probabilité** : Moyenne (nécessite accès physique)

## 4. Scénarios d'attaque implémentés

### 4.1 Attaque naïve
- Injection de frames avec des valeurs arbitraires
- Détectable facilement (fréquence anormale, payload hors range)

### 4.2 Attaque IA Stealth
- Modèle LSTM entraîné sur le trafic normal
- Génère des frames statistiquement similaires au trafic légitime
- Optimise la furtivité via Optuna/CMA-ES
- Objectif : modifier la vitesse affichée sans déclencher l'IDS

## 5. Contre-mesures

| Contre-mesure | Type | Menaces couvertes |
|---------------|------|-------------------|
| ID Whitelist | Règle | Spoofing |
| Frequency Check | Règle | DoS, Spoofing |
| Payload Range | Règle | Tampering |
| Isolation Forest | IA | Spoofing, Tampering |
| Autoencoder | IA | Stealth attacks |
| CUSUM | Statistique | Drift attacks |

## 6. Résiduel

- Les attaques IA stealth avancées peuvent contourner les IDS actuels
- Le bus CAN n'offre aucune authentification native
- Recommandation : CANsec (CAN XL) ou MAC-based authentication
