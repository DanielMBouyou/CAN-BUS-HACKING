"""Modèle LSTM Mimic — Génération de frames CAN stealth (optimisé).

Améliorations par rapport à la v1 :
    1. Multi-Head Self-Attention sur les sorties LSTM
    2. Layer Normalization + Residual connections
    3. Gradient clipping & weight decay (AdamW)
    4. Cosine Annealing LR scheduler avec warmup
    5. Early stopping sur validation split
    6. Teacher forcing avec scheduled sampling
    7. Conditionnement par ID d'ECU (embedding)
    8. Génération multi-pas (autoregressif)

Architecture :
    [Input] → [ID Embed ⊕ Payload] → Proj → LSTM(2 couches, 128)
           → Multi-Head Attention → LayerNorm → FC → Sigmoid
"""

from __future__ import annotations

import copy
import logging
import math
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset, random_split

from canlab.config import ATTACK_CFG, PROJECT_ROOT

logger = logging.getLogger(__name__)

MODELS_DIR = PROJECT_ROOT / "data" / "models"
MODELS_DIR.mkdir(parents=True, exist_ok=True)


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Blocs réutilisables
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━


class TemporalAttention(nn.Module):
    """Multi-Head Self-Attention sur la dimension temporelle.

    Permet au modèle de pondérer dynamiquement les pas de temps
    les plus informatifs au lieu de ne prendre que le dernier.
    """

    def __init__(self, hidden_size: int, n_heads: int = 4, dropout: float = 0.1) -> None:
        super().__init__()
        self.attn = nn.MultiheadAttention(
            embed_dim=hidden_size,
            num_heads=n_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.norm = nn.LayerNorm(hidden_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x : (batch, seq_len, hidden) → (batch, hidden) contexte agrégé."""
        attn_out, _ = self.attn(x, x, x, need_weights=False)
        out = self.norm(x + attn_out)  # residual + LayerNorm
        return out.mean(dim=1)  # mean-pool


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Modèle principal
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━


class CANMimicLSTM(nn.Module):
    """LSTM + Attention pour mimétisme de trafic CAN.

    Changements vs v1 :
    - Embedding d'ID ECU (optionnel) concaténé aux bytes
    - Multi-Head Attention au lieu de last-hidden-only
    - LayerNorm + résiduel entre LSTM et tête FC
    - Dropout cohérent partout
    """

    def __init__(
        self,
        input_size: int = ATTACK_CFG.input_size,
        hidden_size: int = ATTACK_CFG.hidden_size,
        num_layers: int = ATTACK_CFG.num_layers,
        output_size: int | None = None,
        use_attention: bool = ATTACK_CFG.use_attention,
        n_heads: int = ATTACK_CFG.attention_heads,
        dropout: float = ATTACK_CFG.dropout,
        num_ecu_ids: int = 16,
        id_embed_dim: int = 8,
    ) -> None:
        super().__init__()
        if output_size is None:
            output_size = input_size

        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.output_size = output_size
        self.use_attention = use_attention
        self.id_embed_dim = id_embed_dim

        # ── ID Embedding (optionnel) ──
        self.id_embedding = nn.Embedding(num_ecu_ids, id_embed_dim)
        lstm_input = input_size + id_embed_dim

        # ── Couche d'entrée avec projection ──
        self.input_proj = nn.Sequential(
            nn.Linear(lstm_input, hidden_size),
            nn.LayerNorm(hidden_size),
            nn.GELU(),
            nn.Dropout(dropout),
        )

        # ── LSTM ──
        self.lstm = nn.LSTM(
            input_size=hidden_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.lstm_norm = nn.LayerNorm(hidden_size)

        # ── Attention ──
        if use_attention:
            self.attention = TemporalAttention(hidden_size, n_heads, dropout)
        else:
            self.attention = None

        # ── Tête de sortie ──
        self.head = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, hidden_size // 2),
            nn.GELU(),
            nn.Linear(hidden_size // 2, output_size),
            nn.Sigmoid(),
        )

    def forward(
        self, x: torch.Tensor, ecu_ids: torch.Tensor | None = None
    ) -> torch.Tensor:
        """Forward pass.

        Args:
            x: (batch, seq_len, input_size)
            ecu_ids: (batch,) indices d'ECU (optionnel)

        Returns:
            (batch, output_size)
        """
        batch, seq_len, _ = x.shape

        # ── Conditionnement ID ──
        if ecu_ids is not None:
            id_emb = self.id_embedding(ecu_ids).unsqueeze(1).expand(-1, seq_len, -1)
        else:
            id_emb = torch.zeros(batch, seq_len, self.id_embed_dim, device=x.device)
        x_cat = torch.cat([x, id_emb], dim=-1)

        # ── Projection d'entrée ──
        x_proj = self.input_proj(x_cat)

        # ── LSTM + résiduel ──
        lstm_out, _ = self.lstm(x_proj)
        lstm_out = self.lstm_norm(lstm_out + x_proj)

        # ── Agrégation ──
        if self.use_attention and self.attention is not None:
            context = self.attention(lstm_out)
        else:
            context = lstm_out[:, -1, :]

        return self.head(context)

    def generate_frame(
        self,
        sequence: np.ndarray,
        ecu_id: int | None = None,
        device: str = "cpu",
    ) -> np.ndarray:
        """Génère une frame.

        Args:
            sequence: (seq_len, input_size) normalisé [0,1]
            ecu_id: index ECU (optionnel)

        Returns:
            (input_size,) bytes [0, 255]
        """
        self.eval()
        with torch.no_grad():
            x = torch.FloatTensor(sequence).unsqueeze(0).to(device)
            ids = None
            if ecu_id is not None:
                ids = torch.LongTensor([ecu_id]).to(device)
            pred = self(x, ids).squeeze(0).cpu().numpy()
        return (pred * 255).clip(0, 255).astype(np.uint8)

    def generate_sequence(
        self,
        seed_sequence: np.ndarray,
        n_steps: int = 10,
        ecu_id: int | None = None,
        device: str = "cpu",
    ) -> list[np.ndarray]:
        """Génération autorégressif multi-pas (fenêtre glissante)."""
        self.eval()
        seq = seed_sequence.copy().astype(np.float32)
        if seq.max() > 1.0:
            seq = seq / 255.0

        generated: list[np.ndarray] = []
        for _ in range(n_steps):
            frame = self.generate_frame(seq, ecu_id, device)
            generated.append(frame)
            new_row = frame.astype(np.float32) / 255.0
            seq = np.vstack([seq[1:], new_row[np.newaxis, :]])

        return generated


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Préparation des données
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━


def prepare_sequences(
    payloads: np.ndarray,
    seq_len: int = ATTACK_CFG.seq_len,
    ecu_ids: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray] | tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Prépare les séquences d'entraînement.

    Returns:
        (X, Y) ou (X, Y, ids) si ecu_ids fournis
    """
    data = payloads.astype(np.float32) / 255.0

    X, Y = [], []
    id_seqs: list[int] = [] if ecu_ids is not None else []

    for i in range(len(data) - seq_len):
        X.append(data[i : i + seq_len])
        Y.append(data[i + seq_len])
        if ecu_ids is not None:
            id_seqs.append(int(ecu_ids[i + seq_len]))

    X_arr = np.array(X)
    Y_arr = np.array(Y)
    if ecu_ids is not None:
        return X_arr, Y_arr, np.array(id_seqs)
    return X_arr, Y_arr


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# LR Scheduler avec warmup
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━


class CosineWarmupScheduler(torch.optim.lr_scheduler._LRScheduler):
    """Cosine Annealing avec warmup linéaire."""

    def __init__(self, optimizer, warmup_epochs: int, total_epochs: int, last_epoch=-1):
        self.warmup = warmup_epochs
        self.total = total_epochs
        super().__init__(optimizer, last_epoch)

    def get_lr(self):
        if self.last_epoch < self.warmup:
            factor = (self.last_epoch + 1) / max(self.warmup, 1)
        else:
            progress = (self.last_epoch - self.warmup) / max(self.total - self.warmup, 1)
            factor = 0.5 * (1 + math.cos(math.pi * progress))
        return [base * factor for base in self.base_lrs]


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Entraînement
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━


def train_mimic_model(
    payloads: np.ndarray,
    ecu_ids: np.ndarray | None = None,
    model: CANMimicLSTM | None = None,
    epochs: int = ATTACK_CFG.epochs,
    batch_size: int = ATTACK_CFG.batch_size,
    lr: float = ATTACK_CFG.learning_rate,
    device: str = "cpu",
    save_path: Path | None = None,
) -> tuple[CANMimicLSTM, dict]:
    """Entraîne le modèle avec toutes les optimisations.

    Returns:
        model: modèle entraîné (meilleur checkpoint)
        history: dict avec 'train_losses', 'val_losses', 'best_epoch', 'lr_history'
    """
    seq_result = prepare_sequences(payloads, ecu_ids=ecu_ids)
    if ecu_ids is not None:
        X, Y, id_arr = seq_result
    else:
        X, Y = seq_result
        id_arr = None

    logger.info("📊 Données : X=%s, Y=%s", X.shape, Y.shape)

    # ── Datasets ──
    tensors = [torch.FloatTensor(X), torch.FloatTensor(Y)]
    has_ids = id_arr is not None
    if has_ids:
        tensors.append(torch.LongTensor(id_arr))

    full_ds = TensorDataset(*tensors)

    # ── Validation split ──
    n_val = max(1, int(len(full_ds) * ATTACK_CFG.val_split))
    n_train = len(full_ds) - n_val
    train_ds, val_ds = random_split(
        full_ds, [n_train, n_val], generator=torch.Generator().manual_seed(42)
    )
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, drop_last=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)

    logger.info("📊 Split : train=%d, val=%d", n_train, n_val)

    # ── Modèle ──
    if model is None:
        model = CANMimicLSTM()
    model = model.to(device)

    # ── Optimiseur ──
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=lr, weight_decay=ATTACK_CFG.weight_decay
    )

    # ── Scheduler ──
    scheduler = None
    if ATTACK_CFG.scheduler == "cosine":
        scheduler = CosineWarmupScheduler(optimizer, ATTACK_CFG.warmup_epochs, epochs)
    elif ATTACK_CFG.scheduler == "plateau":
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode="min", factor=0.5, patience=5
        )

    criterion = nn.MSELoss()

    # ── Early stopping state ──
    best_val_loss = float("inf")
    best_state = None
    best_epoch = 0
    patience_counter = 0

    history: dict = {
        "train_losses": [],
        "val_losses": [],
        "lr_history": [],
        "best_epoch": 0,
    }

    tf_start = ATTACK_CFG.teacher_forcing_start
    tf_end = ATTACK_CFG.teacher_forcing_end

    for epoch in range(epochs):
        tf_ratio = tf_start - (tf_start - tf_end) * (epoch / max(epochs - 1, 1))

        # ── Train ──
        model.train()
        train_loss = 0.0
        n_train_batches = 0

        for batch_data in train_loader:
            if has_ids:
                x_b, y_b, id_b = batch_data
                x_b, y_b, id_b = x_b.to(device), y_b.to(device), id_b.to(device)
            else:
                x_b, y_b = batch_data
                x_b, y_b = x_b.to(device), y_b.to(device)
                id_b = None

            optimizer.zero_grad()
            pred = model(x_b, id_b)
            loss = criterion(pred, y_b)

            # Teacher forcing noise (encourage robustness)
            if tf_ratio < 1.0 and torch.rand(1).item() > tf_ratio:
                noise = 0.01 * torch.randn_like(pred)
                pred_noisy = model(x_b, id_b)
                loss = loss + 0.1 * criterion(pred_noisy, y_b + noise)

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), ATTACK_CFG.grad_clip_norm)
            optimizer.step()
            train_loss += loss.item()
            n_train_batches += 1

        avg_train = train_loss / max(n_train_batches, 1)
        history["train_losses"].append(avg_train)

        # ── Validation ──
        model.eval()
        val_loss = 0.0
        n_val_batches = 0
        with torch.no_grad():
            for batch_data in val_loader:
                if has_ids:
                    x_b, y_b, id_b = batch_data
                    x_b, y_b, id_b = x_b.to(device), y_b.to(device), id_b.to(device)
                else:
                    x_b, y_b = batch_data
                    x_b, y_b = x_b.to(device), y_b.to(device)
                    id_b = None

                pred = model(x_b, id_b)
                val_loss += criterion(pred, y_b).item()
                n_val_batches += 1

        avg_val = val_loss / max(n_val_batches, 1)
        history["val_losses"].append(avg_val)

        # ── LR Scheduler ──
        current_lr = optimizer.param_groups[0]["lr"]
        history["lr_history"].append(current_lr)
        if isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
            scheduler.step(avg_val)
        elif scheduler is not None:
            scheduler.step()

        if (epoch + 1) % 10 == 0 or epoch == 0:
            logger.info(
                "Epoch %3d/%d — Train: %.6f | Val: %.6f | LR: %.2e | TF: %.2f",
                epoch + 1, epochs, avg_train, avg_val, current_lr, tf_ratio,
            )

        # ── Early stopping ──
        if avg_val < best_val_loss:
            best_val_loss = avg_val
            best_epoch = epoch + 1
            best_state = copy.deepcopy(model.state_dict())
            patience_counter = 0
        else:
            patience_counter += 1

        if ATTACK_CFG.early_stopping and patience_counter >= ATTACK_CFG.patience:
            logger.info(
                "⏹️ Early stopping à epoch %d (patience=%d, best=%d)",
                epoch + 1, ATTACK_CFG.patience, best_epoch,
            )
            break

    # ── Restaurer le meilleur modèle ──
    if best_state is not None:
        model.load_state_dict(best_state)
        logger.info("🏆 Meilleur modèle restauré (epoch %d, val_loss=%.6f)", best_epoch, best_val_loss)

    history["best_epoch"] = best_epoch

    # ── Sauvegarder ──
    if save_path is None:
        save_path = MODELS_DIR / "mimic_lstm.pt"
    save_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "state_dict": model.state_dict(),
            "config": {
                "input_size": model.input_size,
                "hidden_size": model.hidden_size,
                "num_layers": model.num_layers,
                "output_size": model.output_size,
                "use_attention": model.use_attention,
            },
            "history": history,
        },
        save_path,
    )
    logger.info("✅ Modèle sauvegardé : %s", save_path)
    return model, history


def load_mimic_model(
    path: Path | None = None, device: str = "cpu"
) -> CANMimicLSTM:
    """Charge un modèle LSTM pré-entraîné (compatible ancien/nouveau format)."""
    if path is None:
        path = MODELS_DIR / "mimic_lstm.pt"

    data = torch.load(path, map_location=device, weights_only=False)

    if isinstance(data, dict) and "config" in data:
        cfg = data["config"]
        model = CANMimicLSTM(
            input_size=cfg.get("input_size", 8),
            hidden_size=cfg.get("hidden_size", 128),
            num_layers=cfg.get("num_layers", 2),
            output_size=cfg.get("output_size", 8),
            use_attention=cfg.get("use_attention", True),
        )
        model.load_state_dict(data["state_dict"])
    else:
        model = CANMimicLSTM()
        sd = data["state_dict"] if isinstance(data, dict) and "state_dict" in data else data
        model.load_state_dict(sd)

    model.eval()
    logger.info("📦 Modèle chargé : %s", path)
    return model
