"""IDS CUSUM — Détection séquentielle de changement.

Algorithme CUSUM (Cumulative Sum) pour détecter les drifts
progressifs dans le trafic CAN.

Formule :
    S_k = max(0, S_{k-1} + r_k^T Σ^{-1} r_k - δ)

Où :
    r_k = résidu (observation - attendu)
    Σ = matrice de covariance des résidus normaux
    δ = paramètre de sensibilité
    Alerte si S_k > threshold
"""

from __future__ import annotations

import logging
from collections import deque

import numpy as np

from canlab.config import IDS_CFG

logger = logging.getLogger(__name__)


class CUSUMIDS:
    """IDS basé sur l'algorithme CUSUM multivarié."""

    def __init__(
        self,
        delta: float = IDS_CFG.cusum_delta,
        threshold: float = IDS_CFG.cusum_threshold,
        window_size: int = 100,
    ) -> None:
        self.delta = delta
        self.threshold = threshold
        self.window_size = window_size

        # État interne
        self.S: float = 0.0  # Score CUSUM cumulatif
        self._mean: np.ndarray | None = None  # Moyenne des résidus normaux
        self._cov_inv: np.ndarray | None = None  # Inverse de la covariance
        self._calibrated = False
        self._active = False
        self._history: deque[float] = deque(maxlen=1000)
        self._alert_count = 0

    def activate(self) -> None:
        self._active = True
        self.S = 0.0
        logger.info("🛡️ CUSUM IDS activé (δ=%.1f, seuil=%.1f)", self.delta, self.threshold)

    def deactivate(self) -> None:
        self._active = False

    @property
    def is_active(self) -> bool:
        return self._active

    def calibrate(self, features: np.ndarray) -> None:
        """Calibre le CUSUM sur les données normales.

        Calcule la moyenne et la covariance inverse des features.

        Args:
            features: (n_samples, n_features)
        """
        X = np.nan_to_num(features.astype(np.float64), nan=0.0, posinf=1e6, neginf=-1e6)
        self._mean = X.mean(axis=0)

        # Covariance avec régularisation
        cov = np.cov(X, rowvar=False)
        if cov.ndim == 0:
            cov = np.array([[cov]])
        # Régularisation de Tikhonov
        reg = 1e-6 * np.eye(cov.shape[0])
        self._cov_inv = np.linalg.inv(cov + reg)

        self._calibrated = True
        self.S = 0.0
        logger.info(
            "✅ CUSUM calibré sur %d échantillons (dim=%d)",
            len(X),
            X.shape[1],
        )

    def update(self, feature_vector: np.ndarray) -> dict:
        """Met à jour le score CUSUM avec une nouvelle observation.

        Args:
            feature_vector: (n_features,) ou (1, n_features)

        Returns:
            dict avec 'cusum_score', 'is_anomaly', 'residual_norm'
        """
        if not self._active or not self._calibrated:
            return {
                "cusum_score": 0.0,
                "is_anomaly": False,
                "residual_norm": 0.0,
            }

        x = np.nan_to_num(
            feature_vector.astype(np.float64).flatten(),
            nan=0.0,
            posinf=1e6,
            neginf=-1e6,
        )

        # Résidu
        r = x - self._mean

        # Statistique de Mahalanobis : r^T Σ^{-1} r
        mahal = float(r @ self._cov_inv @ r)

        # Mise à jour CUSUM
        self.S = max(0.0, self.S + mahal - self.delta)

        # Détection
        is_anomaly = self.S > self.threshold
        if is_anomaly:
            self._alert_count += 1

        self._history.append(self.S)

        return {
            "cusum_score": self.S,
            "is_anomaly": is_anomaly,
            "residual_norm": mahal,
            "threshold": self.threshold,
        }

    def check_frame(self, feature_vector: np.ndarray) -> dict:
        """Alias pour update()."""
        return self.update(feature_vector)

    def reset(self) -> None:
        """Remet le score CUSUM à zéro."""
        self.S = 0.0
        self._history.clear()
        logger.info("🔄 CUSUM réinitialisé")

    def get_history(self) -> list[float]:
        """Retourne l'historique des scores CUSUM."""
        return list(self._history)

    def get_summary(self) -> dict:
        """Résumé de l'état du CUSUM."""
        return {
            "active": self._active,
            "calibrated": self._calibrated,
            "current_score": self.S,
            "threshold": self.threshold,
            "delta": self.delta,
            "alert_count": self._alert_count,
            "history_length": len(self._history),
        }
