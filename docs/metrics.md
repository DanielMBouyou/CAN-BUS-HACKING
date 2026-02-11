# 📈 Métriques — CAN-Stealth-Attack-AI Lab

## 1. Métriques de détection IDS

### 1.1 Precision

$$
\text{Precision} = \frac{TP}{TP + FP}
$$

Mesure la proportion de détections correctes parmi toutes les alertes.

### 1.2 Recall (Sensibilité)

$$
\text{Recall} = \frac{TP}{TP + FN}
$$

Mesure la proportion d'attaques correctement détectées.

### 1.3 F1-Score

$$
F_1 = 2 \cdot \frac{\text{Precision} \cdot \text{Recall}}{\text{Precision} + \text{Recall}}
$$

Moyenne harmonique de Precision et Recall.

### 1.4 False Positive Rate (FPR)

$$
\text{FPR} = \frac{FP}{FP + TN}
$$

Taux de fausses alertes. Critique en automobile (fausses alertes = freinage intempestif).

### 1.5 Detection Delay

$$
\text{Delay} = t_{\text{detection}} - t_{\text{attack\_start}}
$$

Temps entre le début de l'attaque et sa détection (en ms ou en nombre de frames).

---

## 2. Métriques d'attaque

### 2.1 Attack Success Rate (ASR)

$$
\text{ASR} = \frac{\text{Attaques non détectées}}{\text{Total attaques}}
$$

### 2.2 Speed Delta

$$
\Delta v = |v_{\text{affichée}} - v_{\text{réelle}}|
$$

Écart entre la vitesse affichée au conducteur et la vitesse réelle du véhicule.

### 2.3 Stealth Score

$$
\text{Stealth} = 1 - \max(\text{IDS\_scores})
$$

Score composite de furtivité (1 = indétectable, 0 = détecté immédiatement).

---

## 3. Résultats attendus

| Méthode IDS | Attaque naïve | Attaque IA stealth |
|-------------|---------------|---------------------|
| Rules | F1 > 0.95 | F1 < 0.30 |
| Isolation Forest | F1 > 0.90 | F1 ~ 0.50 |
| Autoencoder | F1 > 0.85 | F1 ~ 0.60 |
| CUSUM | F1 > 0.80 | F1 ~ 0.70 |
| Ensemble | F1 > 0.98 | F1 ~ 0.80 |

---

## 4. Visualisations

- **Confusion Matrix** par méthode IDS
- **ROC Curve** : TPR vs FPR
- **Timeline** : scores IDS en temps réel
- **Histogrammes** : distribution des features normal vs attaque
- **CUSUM Chart** : évolution du score cumulatif
