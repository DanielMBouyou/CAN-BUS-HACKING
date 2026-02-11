# 🛡️ CAN-Stealth-Attack-AI Lab

> Laboratoire complet d'attaque furtive IA sur réseau CAN automobile simulé.

![Python](https://img.shields.io/badge/Python-3.11-blue)
![License](https://img.shields.io/badge/License-MIT-green)
![Docker](https://img.shields.io/badge/Docker-Ready-blue)

---

## 📋 Objectif

Construire un laboratoire de cybersécurité embarquée permettant :

1. **Simulation** d'un réseau CAN automobile (vcan0)
2. **Génération** de trafic ECU réaliste (Engine, ABS, Steering, Cluster)
3. **Injection** d'attaques CAN (naïves + IA stealth via LSTM)
4. **Détection** par IDS : règles classiques, Isolation Forest, Autoencoder, CUSUM
5. **Démonstration live** avec dashboard Streamlit + API FastAPI

---

## 🏗️ Architecture

```
[ ECU Simulators ] ---> [ Virtual CAN (vcan0) ] ---> [ Logger ]
                                   |
                                   v
                           [ Feature Pipeline ]
                                   |
                    --------------------------------
                    |                              |
            [ IA Attacker ]                [ IDS System ]
                    |                              |
                    --------------------------------
                                   |
                                   v
                           [ API + Dashboard ]
```

---

## ⚙️ Prérequis

### Windows 11 + WSL2 (recommandé)

```powershell
wsl --install -d Ubuntu
```

### Dans WSL2 Ubuntu

```bash
sudo apt update && sudo apt install -y can-utils python3 python3-pip python3-venv git build-essential

# Setup Virtual CAN
sudo modprobe vcan
sudo ip link add dev vcan0 type vcan
sudo ip link set up vcan0

# Vérifier
ip link show vcan0
```

---

## 🚀 Installation

```bash
# Cloner le repo
git clone https://github.com/DanielMBouyou/CAN-BUS-HACKING.git
cd CAN-BUS-HACKING

# Créer environnement virtuel
python3 -m venv .venv
source .venv/bin/activate

# Installer le projet
pip install -e ".[dev]"
```

### Ou via Docker

```bash
cd docker
docker compose up --build
```

---

## 🎮 Utilisation rapide

### 1. Lancer la simulation CAN

```bash
python -m canlab.sim.bus
```

### 2. Collecter et traiter les données

```bash
python -m canlab.data.ingest
python -m canlab.data.features
```

### 3. Lancer l'IDS

```bash
python -m canlab.ids.rules
```

### 4. Lancer l'attaque IA stealth

```bash
python -m canlab.attack.injector
```

### 5. Dashboard live

```bash
# Terminal 1 : API
uvicorn canlab.api.main:app --host 0.0.0.0 --port 8000

# Terminal 2 : Dashboard
streamlit run src/canlab/ui/app_streamlit.py
```

---

## 📊 ECUs Simulés

| ECU      | ID (hex) | Fréquence | Payload       |
|----------|----------|-----------|---------------|
| Engine   | 0x100    | 10 ms     | RPM           |
| ABS      | 0x110    | 20 ms     | Wheel speed   |
| Steering | 0x120    | 50 ms     | Angle         |
| Cluster  | 0x130    | 100 ms    | Speed display |

---

## 📈 Métriques

- Precision / Recall / F1
- False Positive Rate
- Detection Delay
- Attack Success Rate

---

## 📁 Structure du projet

```
can-stealth-attack-ai/
├── README.md
├── pyproject.toml
├── docker/
│   ├── Dockerfile
│   └── docker-compose.yml
├── docs/
│   ├── threat_model.md
│   ├── architecture.md
│   ├── demo_script.md
│   └── metrics.md
├── src/
│   └── canlab/
│       ├── config.py
│       ├── sim/          # Simulation ECU + bus
│       ├── data/         # Ingestion + feature engineering
│       ├── attack/       # IA attaquante (LSTM + Optuna)
│       ├── ids/          # Systèmes IDS
│       ├── api/          # FastAPI
│       └── ui/           # Streamlit dashboard
├── tests/
└── data/
    ├── raw/
    ├── processed/
    └── features/
```

---

## 🧪 Tests

```bash
pytest tests/ -v
```

---

## 📜 Licence

MIT

---

## 👤 Auteur

**DanielMBouyou** — [GitHub](https://github.com/DanielMBouyou)
