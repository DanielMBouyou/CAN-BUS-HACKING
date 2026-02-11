"""Modèle LSTM Mimic — Génération de frames CAN stealth.

Architecture :
    - LSTM 2 couches, hidden_size=128
    - Entrée : séquence de 50 frames (payload bytes)
    - Sortie : prochaine frame candidate

Le modèle apprend le pattern temporel du trafic normal
et génère des frames statistiquement similaires.
"""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from canlab.config import ATTACK_CFG, PROJECT_ROOT

logger = logging.getLogger(__name__)

MODELS_DIR = PROJECT_ROOT / "data" / "models"
MODELS_DIR.mkdir(parents=True, exist_ok=True)


class CANMimicLSTM(nn.Module):
    """LSTM pour mimétisme de trafic CAN.

    Apprend à prédire la prochaine frame CAN étant donnée
    une séquence de frames précédentes.
    """

    def __init__(
        self,
        input_size: int = ATTACK_CFG.input_size,
        hidden_size: int = ATTACK_CFG.hidden_size,
        num_layers: int = ATTACK_CFG.num_layers,
        output_size: int | None = None,
    ) -> None:
        super().__init__()
        if output_size is None:
            output_size = input_size

        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.output_size = output_size

        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=0.2 if num_layers > 1 else 0.0,
        )
        self.fc = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.Linear(hidden_size // 2, output_size),
            nn.Sigmoid(),  # Sortie entre 0 et 1 (bytes normalisés)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Args:
            x: (batch, seq_len, input_size) — séquence de frames normalisées

        Returns:
            (batch, output_size) — prochaine frame prédite
        """
        lstm_out, _ = self.lstm(x)
        # Prendre la dernière sortie temporelle
        last_hidden = lstm_out[:, -1, :]
        return self.fc(last_hidden)

    def generate_frame(
        self, sequence: np.ndarray, device: str = "cpu"
    ) -> np.ndarray:
        """Génère une frame à partir d'une séquence d'entrée.

        Args:
            sequence: (seq_len, input_size) frames normalisées [0, 1]

        Returns:
            (input_size,) frame générée [0, 255] bytes
        """
        self.eval()
        with torch.no_grad():
            x = torch.FloatTensor(sequence).unsqueeze(0).to(device)
            pred = self(x).squeeze(0).cpu().numpy()
        # Dénormaliser → bytes
        return (pred * 255).clip(0, 255).astype(np.uint8)


def prepare_sequences(
    payloads: np.ndarray, seq_len: int = ATTACK_CFG.seq_len
) -> tuple[np.ndarray, np.ndarray]:
    """Prépare les séquences d'entraînement.

    Args:
        payloads: (n_frames, 8) bytes des payloads
        seq_len: longueur de la séquence

    Returns:
        X: (n_samples, seq_len, 8) séquences d'entrée
        Y: (n_samples, 8) frames cibles
    """
    # Normaliser entre 0 et 1
    data = payloads.astype(np.float32) / 255.0

    X, Y = [], []
    for i in range(len(data) - seq_len):
        X.append(data[i : i + seq_len])
        Y.append(data[i + seq_len])

    return np.array(X), np.array(Y)


def train_mimic_model(
    payloads: np.ndarray,
    model: CANMimicLSTM | None = None,
    epochs: int = ATTACK_CFG.epochs,
    batch_size: int = ATTACK_CFG.batch_size,
    lr: float = ATTACK_CFG.learning_rate,
    device: str = "cpu",
    save_path: Path | None = None,
) -> tuple[CANMimicLSTM, list[float]]:
    """Entraîne le modèle LSTM sur le trafic CAN normal.

    Returns:
        model: modèle entraîné
        losses: historique des pertes
    """
    X, Y = prepare_sequences(payloads)
    logger.info(
        "📊 Données entraînement : X=%s, Y=%s", X.shape, Y.shape
    )

    dataset = TensorDataset(
        torch.FloatTensor(X),
        torch.FloatTensor(Y),
    )
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    if model is None:
        model = CANMimicLSTM()
    model = model.to(device)
    model.train()

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.MSELoss()
    losses: list[float] = []

    for epoch in range(epochs):
        epoch_loss = 0.0
        n_batches = 0
        for x_batch, y_batch in loader:
            x_batch = x_batch.to(device)
            y_batch = y_batch.to(device)

            optimizer.zero_grad()
            pred = model(x_batch)
            loss = criterion(pred, y_batch)
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()
            n_batches += 1

        avg_loss = epoch_loss / max(n_batches, 1)
        losses.append(avg_loss)
        if (epoch + 1) % 10 == 0:
            logger.info("Epoch %d/%d — Loss: %.6f", epoch + 1, epochs, avg_loss)

    if save_path is None:
        save_path = MODELS_DIR / "mimic_lstm.pt"
    save_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), save_path)
    logger.info("✅ Modèle sauvegardé : %s", save_path)

    return model, losses


def load_mimic_model(
    path: Path | None = None, device: str = "cpu"
) -> CANMimicLSTM:
    """Charge un modèle LSTM pré-entraîné."""
    if path is None:
        path = MODELS_DIR / "mimic_lstm.pt"
    model = CANMimicLSTM()
    model.load_state_dict(torch.load(path, map_location=device, weights_only=True))
    model.eval()
    logger.info("📦 Modèle chargé : %s", path)
    return model
