"""CAN-Stealth-Attack-AI Lab — Configuration centrale."""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


# ──────────────────────────────────────────────
# Chemins
# ──────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parents[2]  # racine du repo
DATA_DIR = PROJECT_ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"
FEATURES_DIR = DATA_DIR / "features"

for _d in (RAW_DIR, PROCESSED_DIR, FEATURES_DIR):
    _d.mkdir(parents=True, exist_ok=True)


# ──────────────────────────────────────────────
# Interface CAN
# ──────────────────────────────────────────────
CAN_INTERFACE = os.getenv("CAN_INTERFACE", "vcan0")
CAN_BUSTYPE = os.getenv("CAN_BUSTYPE", "socketcan")

# Fallback pour Windows (mode simulation sans SocketCAN réel)
USE_VIRTUAL_BUS = os.getenv("USE_VIRTUAL_BUS", "false").lower() == "true"
if USE_VIRTUAL_BUS:
    CAN_BUSTYPE = "virtual"


# ──────────────────────────────────────────────
# Définition des ECUs
# ──────────────────────────────────────────────
@dataclass(frozen=True)
class ECUDef:
    """Définition d'une ECU simulée."""

    name: str
    arb_id: int
    period_ms: float
    dlc: int = 8
    description: str = ""


ECU_ENGINE = ECUDef(
    name="Engine", arb_id=0x100, period_ms=10, description="RPM moteur"
)
ECU_ABS = ECUDef(
    name="ABS", arb_id=0x110, period_ms=20, description="Vitesse roues"
)
ECU_STEER = ECUDef(
    name="Steering", arb_id=0x120, period_ms=50, description="Angle volant"
)
ECU_CLUSTER = ECUDef(
    name="Cluster", arb_id=0x130, period_ms=100, description="Affichage vitesse"
)

ALL_ECUS: list[ECUDef] = [ECU_ENGINE, ECU_ABS, ECU_STEER, ECU_CLUSTER]
ECU_BY_ID: dict[int, ECUDef] = {e.arb_id: e for e in ALL_ECUS}
VALID_IDS: set[int] = {e.arb_id for e in ALL_ECUS}


# ──────────────────────────────────────────────
# Paramètres physiques
# ──────────────────────────────────────────────
@dataclass
class PhysicsParams:
    """Paramètres du modèle dynamique simplifié."""

    # RPM_{k+1} = RPM_k + alpha*(throttle - load) + noise
    alpha: float = 0.5
    rpm_min: float = 800.0
    rpm_max: float = 7000.0
    throttle_default: float = 0.3
    load_default: float = 0.2
    noise_std: float = 10.0
    # v = (2*pi*R / 60) * RPM
    wheel_radius_m: float = 0.32
    # Steering
    steer_amplitude_deg: float = 45.0
    steer_freq_hz: float = 0.1


PHYSICS = PhysicsParams()


# ──────────────────────────────────────────────
# Feature engineering
# ──────────────────────────────────────────────
@dataclass
class FeatureConfig:
    """Configuration du pipeline de features."""

    window_size: int = 50
    rolling_window: int = 20
    entropy_bins: int = 16


FEATURES_CFG = FeatureConfig()


# ──────────────────────────────────────────────
# IA Attaquante (LSTM)
# ──────────────────────────────────────────────
@dataclass
class AttackModelConfig:
    """Config du modèle LSTM attaquant."""

    input_size: int = 8  # taille payload CAN
    hidden_size: int = 128
    num_layers: int = 2
    seq_len: int = 50
    learning_rate: float = 1e-3
    epochs: int = 100
    batch_size: int = 64
    # ── Attention ──
    use_attention: bool = True
    attention_heads: int = 4
    # ── Régularisation ──
    dropout: float = 0.2
    grad_clip_norm: float = 1.0
    weight_decay: float = 1e-5
    # ── Scheduler ──
    scheduler: str = "cosine"  # "cosine" | "plateau" | "none"
    warmup_epochs: int = 5
    # ── Early stopping ──
    early_stopping: bool = True
    patience: int = 10
    val_split: float = 0.15
    # ── Teacher forcing ──
    teacher_forcing_start: float = 1.0
    teacher_forcing_end: float = 0.0
    # ── Optimiseur furtivité ──
    optim_n_trials: int = 200
    optim_temporal_weight: float = 0.3
    # ── Injection ──
    timing_jitter_pct: float = 0.10  # ±10 % de jitter sur le timing
    drift_smoothing_window: int = 20


ATTACK_CFG = AttackModelConfig()


# ──────────────────────────────────────────────
# IDS
# ──────────────────────────────────────────────
@dataclass
class IDSConfig:
    """Configuration de l'IDS."""

    # Règles classiques
    max_freq_tolerance: float = 1.5  # multiplicateur de la fréquence nominale
    # Isolation Forest
    if_contamination: float = 0.05
    if_n_estimators: int = 200
    # Autoencoder
    ae_input_dim: int = 5
    ae_hidden_dim: int = 64
    ae_latent_dim: int = 32
    ae_threshold_percentile: float = 95.0
    ae_epochs: int = 30
    ae_lr: float = 1e-3
    # CUSUM
    cusum_delta: float = 4.0
    cusum_threshold: float = 15.0


IDS_CFG = IDSConfig()


# ──────────────────────────────────────────────
# API / Dashboard
# ──────────────────────────────────────────────
API_HOST = os.getenv("API_HOST", "0.0.0.0")
API_PORT = int(os.getenv("API_PORT", "8000"))
DASHBOARD_PORT = int(os.getenv("DASHBOARD_PORT", "8501"))


# ──────────────────────────────────────────────
# Logging
# ──────────────────────────────────────────────
LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO")
