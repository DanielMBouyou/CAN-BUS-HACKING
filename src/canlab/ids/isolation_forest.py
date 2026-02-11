"""IDS Isolation Forest — Détection d'anomalies non supervisée.

Utilise sklearn.ensemble.IsolationForest sur les features CAN
pour détecter les frames anormales.
"""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest as SklearnIF
from sklearn.preprocessing import StandardScaler

from canlab.config import IDS_CFG, PROJECT_ROOT

logger = logging.getLogger(__name__)

MODELS_DIR = PROJECT_ROOT / "data" / "models"
MODELS_DIR.mkdir(parents=True, exist_ok=True)


class IsolationForestIDS:
    """IDS basé sur Isolation Forest.

    Entraîné sur le trafic normal, détecte les anomalies
    par isolation des points atypiques.
    """

    def __init__(
        self,
        contamination: float = IDS_CFG.if_contamination,
        n_estimators: int = IDS_CFG.if_n_estimators,
    ) -> None:
        self.contamination = contamination
        self.n_estimators = n_estimators
        self.model: SklearnIF | None = None
        self.scaler = StandardScaler()
        self._fitted = False
        self._active = False
        self._feature_names = [
            "delta_t",
            "id_freq",
            "entropy",
            "payload_rolling_mean",
            "payload_rolling_std",
            "burstiness",
        ]

    def activate(self) -> None:
        self._active = True
        logger.info("🛡️ Isolation Forest IDS activé")

    def deactivate(self) -> None:
        self._active = False

    @property
    def is_active(self) -> bool:
        return self._active

    def fit(self, features: np.ndarray | pd.DataFrame) -> None:
        """Entraîne le modèle sur les données normales.

        Args:
            features: matrice (n_samples, n_features)
        """
        if isinstance(features, pd.DataFrame):
            available = [c for c in self._feature_names if c in features.columns]
            X = features[available].values
        else:
            X = features

        # Nettoyer NaN / Inf
        X = np.nan_to_num(X, nan=0.0, posinf=1e6, neginf=-1e6)

        self.scaler.fit(X)
        X_scaled = self.scaler.transform(X)

        self.model = SklearnIF(
            n_estimators=self.n_estimators,
            contamination=self.contamination,
            random_state=42,
            n_jobs=-1,
        )
        self.model.fit(X_scaled)
        self._fitted = True
        logger.info(
            "✅ Isolation Forest entraîné sur %d échantillons", len(X)
        )

    def predict(self, features: np.ndarray) -> np.ndarray:
        """Prédit les anomalies.

        Returns:
            Array de -1 (anomalie) ou 1 (normal)
        """
        if not self._fitted or self.model is None:
            raise RuntimeError("Modèle non entraîné. Appelez fit() d'abord.")

        X = np.nan_to_num(features, nan=0.0, posinf=1e6, neginf=-1e6)
        if X.ndim == 1:
            X = X.reshape(1, -1)
        X_scaled = self.scaler.transform(X)
        return self.model.predict(X_scaled)

    def score_samples(self, features: np.ndarray) -> np.ndarray:
        """Score d'anomalie continu (plus négatif = plus anormal).

        Returns:
            Array de scores (float)
        """
        if not self._fitted or self.model is None:
            raise RuntimeError("Modèle non entraîné.")

        X = np.nan_to_num(features, nan=0.0, posinf=1e6, neginf=-1e6)
        if X.ndim == 1:
            X = X.reshape(1, -1)
        X_scaled = self.scaler.transform(X)
        return self.model.score_samples(X_scaled)

    def check_frame(self, feature_vector: np.ndarray) -> dict:
        """Vérifie une frame.

        Returns:
            dict avec 'is_anomaly', 'score', 'threshold'
        """
        if not self._active or not self._fitted:
            return {"is_anomaly": False, "score": 0.0, "threshold": 0.0}

        pred = self.predict(feature_vector)
        score = self.score_samples(feature_vector)

        return {
            "is_anomaly": bool(pred[0] == -1),
            "score": float(score[0]),
            "prediction": int(pred[0]),
        }

    def save(self, path: Path | None = None) -> None:
        """Sauvegarde le modèle."""
        import joblib

        if path is None:
            path = MODELS_DIR / "isolation_forest.joblib"
        path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump({"model": self.model, "scaler": self.scaler}, path)
        logger.info("💾 Isolation Forest sauvegardé : %s", path)

    def load(self, path: Path | None = None) -> None:
        """Charge un modèle sauvegardé."""
        import joblib

        if path is None:
            path = MODELS_DIR / "isolation_forest.joblib"
        data = joblib.load(path)
        self.model = data["model"]
        self.scaler = data["scaler"]
        self._fitted = True
        logger.info("📦 Isolation Forest chargé : %s", path)
