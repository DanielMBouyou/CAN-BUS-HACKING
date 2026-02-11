# 🏗️ Architecture — CAN-Stealth-Attack-AI Lab

## Vue d'ensemble

```
┌─────────────────────────────────────────────────────────┐
│                    SIMULATION LAYER                       │
│  ┌──────────┐ ┌──────────┐ ┌──────────┐ ┌──────────┐   │
│  │ Engine   │ │   ABS    │ │ Steering │ │ Cluster  │   │
│  │ ECU 0x100│ │ ECU 0x110│ │ ECU 0x120│ │ ECU 0x130│   │
│  └────┬─────┘ └────┬─────┘ └────┬─────┘ └────┬─────┘   │
│       └─────────────┴─────────────┴─────────────┘        │
│                         │                                 │
│                    [ vcan0 ]                              │
└─────────────────────┬───────────────────────────────────┘
                      │
┌─────────────────────┴───────────────────────────────────┐
│                    DATA LAYER                             │
│  ┌──────────┐      ┌────────────────┐                    │
│  │ Ingestor │ ───► │ Feature Engine │                    │
│  │ (candump)│      │  Δt, freq,     │                    │
│  └──────────┘      │  entropy, μ, σ │                    │
│                    └───────┬────────┘                    │
└────────────────────────────┼────────────────────────────┘
                             │
          ┌──────────────────┼──────────────────┐
          │                  │                  │
┌─────────┴────────┐ ┌──────┴──────┐ ┌─────────┴────────┐
│  ATTACK LAYER    │ │  IDS LAYER  │ │  PRESENTATION    │
│  ┌─────────────┐ │ │ ┌─────────┐ │ │  ┌────────────┐  │
│  │ LSTM Mimic  │ │ │ │ Rules   │ │ │  │ FastAPI    │  │
│  │ Model       │ │ │ ├─────────┤ │ │  │ REST + WS  │  │
│  ├─────────────┤ │ │ │ IsoForst│ │ │  ├────────────┤  │
│  │ Optuna      │ │ │ ├─────────┤ │ │  │ Streamlit  │  │
│  │ Optimizer   │ │ │ │ AutoEnc │ │ │  │ Dashboard  │  │
│  ├─────────────┤ │ │ ├─────────┤ │ │  └────────────┘  │
│  │ Injector    │ │ │ │ CUSUM   │ │ │                   │
│  └─────────────┘ │ │ └─────────┘ │ │                   │
└──────────────────┘ └─────────────┘ └───────────────────┘
```

## Composants

### 1. Simulation Layer (`canlab.sim`)
- **bus.py** : Orchestre le bus CAN virtuel, lance les ECUs
- **ecu_engine.py** : Simule RPM avec modèle dynamique
- **ecu_abs.py** : Simule vitesse roues (dérivée du RPM)
- **ecu_steer.py** : Simule angle volant (sinusoïdal)
- **ecu_cluster.py** : Agrège et affiche la vitesse

### 2. Data Layer (`canlab.data`)
- **ingest.py** : Capture et parse les logs CAN (candump format)
- **features.py** : Calcule les features statistiques par fenêtre temporelle

### 3. Attack Layer (`canlab.attack`)
- **mimic_model.py** : LSTM 2 couches entraîné sur trafic normal
- **optimizer.py** : Optimisation furtivité via Optuna/CMA-ES
- **injector.py** : Injecte les frames forgées sur le bus

### 4. IDS Layer (`canlab.ids`)
- **rules.py** : Vérification fréquence, payload range, ID whitelist
- **isolation_forest.py** : Détection anomalies non supervisée
- **autoencoder.py** : Reconstruction error comme score anomalie
- **cusum.py** : Détection séquentielle de changement

### 5. Presentation Layer (`canlab.api` + `canlab.ui`)
- **main.py** : API REST + WebSocket pour streaming
- **app_streamlit.py** : Dashboard interactif temps réel

## Flux de données

1. ECUs → frames CAN → vcan0
2. Ingestor capture → raw logs
3. Feature pipeline → vecteurs [Δt, freq, entropy, mean, std]
4. IDS consomme les features → scores anomalie
5. Attacker observe le trafic → génère frames stealth
6. API expose l'état → Dashboard affiche

## Communication inter-modules

- **CAN Bus** : python-can (SocketCAN ou virtual)
- **Données** : Parquet files via pyarrow
- **API** : HTTP REST + WebSocket
- **Interne** : Queues Python (asyncio.Queue)
