# 🎬 Script de Démonstration — CAN-Stealth-Attack-AI Lab

## Prérequis

1. WSL2 Ubuntu avec vcan0 configuré
2. Environnement Python installé
3. Tous les modules fonctionnels

---

## Étape 1 : Lancer le simulateur CAN

```bash
# Terminal 1
source .venv/bin/activate
python -m canlab.sim.bus
```

**Résultat attendu** : Les 4 ECUs envoient des frames à leur fréquence nominale.

---

## Étape 2 : Collecter le trafic normal (30 secondes)

```bash
# Terminal 2
candump -L vcan0 > data/raw/normal_traffic.log
# Attendre 30 secondes, puis Ctrl+C
```

---

## Étape 3 : Traiter les données

```bash
python -m canlab.data.ingest --input data/raw/normal_traffic.log
python -m canlab.data.features
```

---

## Étape 4 : Lancer l'API et le Dashboard

```bash
# Terminal 3 : API
uvicorn canlab.api.main:app --host 0.0.0.0 --port 8000

# Terminal 4 : Dashboard
streamlit run src/canlab/ui/app_streamlit.py
```

Ouvrir http://localhost:8501 dans le navigateur.

---

## Étape 5 : Activer l'IDS

Via le dashboard ou via l'API :

```bash
curl http://localhost:8000/ids/status
```

**Observer** : Dashboard montre le trafic normal, scores IDS bas.

---

## Étape 6 : Lancer une attaque naïve

```bash
# Terminal 5
curl -X POST http://localhost:8000/attack/start -H "Content-Type: application/json" \
    -d '{"mode": "naive", "target_id": "0x130", "target_speed": 200}'
```

**Observer** :
- ⚠️ IDS détecte immédiatement (fréquence anormale, payload hors range)
- Dashboard affiche alertes rouges
- Métriques : haute précision de détection

---

## Étape 7 : Stopper l'attaque naïve

```bash
curl -X POST http://localhost:8000/attack/stop
```

---

## Étape 8 : Lancer l'attaque IA Stealth

```bash
curl -X POST http://localhost:8000/attack/start -H "Content-Type: application/json" \
    -d '{"mode": "stealth", "target_id": "0x130", "target_speed": 200}'
```

**Observer** :
- 🟢 IDS règles : aucune alerte (timing et payload normaux)
- 🟡 Isolation Forest : score limite
- 🟠 Autoencoder : légère augmentation
- La vitesse affichée diverge de la vitesse réelle
- CUSUM détecte le drift progressif

---

## Étape 9 : Comparer les résultats

Dans le Dashboard :
- **Graphe "Vitesse réelle vs affichée"** : écart visible
- **Scores IDS** : comparaison naïve vs stealth
- **Métriques** : Precision, Recall, F1, FPR pour chaque méthode

---

## Étape 10 : Arrêter tout

```bash
curl -X POST http://localhost:8000/attack/stop
# Ctrl+C sur tous les terminaux
```

---

## Points clés à démontrer

1. **CAN est vulnérable par design** : pas d'authentification
2. **Attaques naïves** : faciles à détecter
3. **Attaques IA stealth** : contournent les IDS classiques
4. **Défense en profondeur** : combiner règles + ML + statistique
5. **CUSUM** : détecte les drifts que l'IA ne peut pas masquer
