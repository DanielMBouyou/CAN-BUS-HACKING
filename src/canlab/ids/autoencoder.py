"""IDS Autoencoder PyTorch — Détection par erreur de reconstruction.

Architecture :
    Input(n_features) → Encoder(64 → 32) → Decoder(32 → 64) → Output(n_features)

Score d'anomalie :
    Score = ||x - x_hat||
"""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from canlab.config import IDS_CFG, PROJECT_ROOT

logger = logging.getLogger(__name__)

MODELS_DIR = PROJECT_ROOT / "data" / "models"
MODELS_DIR.mkdir(parents=True, exist_ok=True)


class CANAutoencoder(nn.Module):
    """Autoencoder pour détection d'anomalies CAN.

    Entraîné sur le trafic normal, les frames anormales
    auront une erreur de reconstruction élevée.
    """

    def __init__(
        self,
        input_dim: int = IDS_CFG.ae_input_dim,
        hidden_dim: int = IDS_CFG.ae_hidden_dim,
        latent_dim: int = IDS_CFG.ae_latent_dim,
    ) -> None:
        super().__init__()
        self.input_dim = input_dim

        # Encoder
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.BatchNorm1d(hidden_dim),
            nn.Linear(hidden_dim, latent_dim),
            nn.ReLU(),
        )

        # Decoder
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.ReLU(),
            nn.BatchNorm1d(hidden_dim),
            nn.Linear(hidden_dim, input_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward : encode → decode."""
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """Retourne la représentation latente."""
        return self.encoder(x)

    def reconstruction_error(self, x: torch.Tensor) -> torch.Tensor:
        """Calcule l'erreur de reconstruction par échantillon."""
        x_hat = self.forward(x)
        return torch.mean((x - x_hat) ** 2, dim=1)


class AutoencoderIDS:
    """IDS basé sur l'Autoencoder PyTorch."""

    def __init__(
        self,
        input_dim: int = IDS_CFG.ae_input_dim,
        threshold_percentile: float = IDS_CFG.ae_threshold_percentile,
        epochs: int = IDS_CFG.ae_epochs,
        lr: float = IDS_CFG.ae_lr,
        device: str = "cpu",
    ) -> None:
        self.input_dim = input_dim
        self.threshold_percentile = threshold_percentile
        self.epochs = epochs
        self.lr = lr
        self.device = device
        self.model = CANAutoencoder(input_dim=input_dim)
        self.threshold: float = 0.0
        self._fitted = False
        self._active = False
        self._mean: np.ndarray | None = None
        self._std: np.ndarray | None = None

    def activate(self) -> None:
        self._active = True
        logger.info("🛡️ Autoencoder IDS activé")

    def deactivate(self) -> None:
        self._active = False

    @property
    def is_active(self) -> bool:
        return self._active

    def _normalize(self, X: np.ndarray) -> np.ndarray:
        """Normalisation Z-score."""
        if self._mean is None:
            self._mean = X.mean(axis=0)
            self._std = X.std(axis=0) + 1e-8
        return (X - self._mean) / self._std

    def fit(self, features: np.ndarray, batch_size: int = 64) -> list[float]:
        """Entraîne l'autoencoder sur les données normales.

        Returns:
            Historique des pertes
        """
        X = np.nan_to_num(features.astype(np.float32), nan=0.0, posinf=1e6, neginf=-1e6)

        # Ajuster input_dim si nécessaire
        if X.shape[1] != self.input_dim:
            self.input_dim = X.shape[1]
            self.model = CANAutoencoder(input_dim=self.input_dim)

        X_norm = self._normalize(X)
        dataset = TensorDataset(torch.FloatTensor(X_norm))
        loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

        self.model = self.model.to(self.device)
        self.model.train()
        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr)
        criterion = nn.MSELoss()

        losses: list[float] = []
        for epoch in range(self.epochs):
            epoch_loss = 0.0
            n = 0
            for (batch,) in loader:
                batch = batch.to(self.device)
                optimizer.zero_grad()
                output = self.model(batch)
                loss = criterion(output, batch)
                loss.backward()
                optimizer.step()
                epoch_loss += loss.item()
                n += 1

            avg = epoch_loss / max(n, 1)
            losses.append(avg)
            if (epoch + 1) % 10 == 0:
                logger.info("AE Epoch %d/%d — Loss: %.6f", epoch + 1, self.epochs, avg)

        # Calculer le seuil sur les données d'entraînement
        self.model.eval()
        with torch.no_grad():
            X_tensor = torch.FloatTensor(X_norm).to(self.device)
            errors = self.model.reconstruction_error(X_tensor).cpu().numpy()

        self.threshold = float(np.percentile(errors, self.threshold_percentile))
        self._fitted = True
        logger.info(
            "✅ Autoencoder entraîné — Seuil: %.6f (P%.0f)",
            self.threshold,
            self.threshold_percentile,
        )
        return losses

    def score(self, features: np.ndarray) -> np.ndarray:
        """Calcule le score d'anomalie (erreur de reconstruction).

        Returns:
            Array de scores (float) — plus élevé = plus anormal
        """
        if not self._fitted:
            raise RuntimeError("Modèle non entraîné.")

        X = np.nan_to_num(features.astype(np.float32), nan=0.0, posinf=1e6, neginf=-1e6)
        if X.ndim == 1:
            X = X.reshape(1, -1)
        X_norm = (X - self._mean) / self._std

        self.model.eval()
        with torch.no_grad():
            X_tensor = torch.FloatTensor(X_norm).to(self.device)
            errors = self.model.reconstruction_error(X_tensor).cpu().numpy()
        return errors

    def check_frame(self, feature_vector: np.ndarray) -> dict:
        """Vérifie une frame.

        Returns:
            dict avec 'is_anomaly', 'score', 'threshold'
        """
        if not self._active or not self._fitted:
            return {"is_anomaly": False, "score": 0.0, "threshold": self.threshold}

        errors = self.score(feature_vector)
        return {
            "is_anomaly": bool(errors[0] > self.threshold),
            "score": float(errors[0]),
            "threshold": self.threshold,
        }

    def save(self, path: Path | None = None) -> None:
        """Sauvegarde le modèle."""
        if path is None:
            path = MODELS_DIR / "autoencoder.pt"
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "state_dict": self.model.state_dict(),
                "threshold": self.threshold,
                "input_dim": self.input_dim,
                "mean": self._mean,
                "std": self._std,
            },
            path,
        )
        logger.info("💾 Autoencoder sauvegardé : %s", path)

    def load(self, path: Path | None = None) -> None:
        """Charge un modèle sauvegardé."""
        if path is None:
            path = MODELS_DIR / "autoencoder.pt"
        data = torch.load(path, map_location=self.device, weights_only=False)
        self.input_dim = data["input_dim"]
        self.model = CANAutoencoder(input_dim=self.input_dim)
        self.model.load_state_dict(data["state_dict"])
        self.threshold = data["threshold"]
        self._mean = data["mean"]
        self._std = data["std"]
        self._fitted = True
        logger.info("📦 Autoencoder chargé : %s", path)
