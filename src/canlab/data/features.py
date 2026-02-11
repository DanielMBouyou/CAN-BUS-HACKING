"""Feature Engineering pour le trafic CAN.

Calcule les features statistiques par fenêtre temporelle :
- Δt inter-frame
- Fréquence par ID
- Entropie du payload
- Rolling mean / variance
- Burstiness

Vecteur de features :
    x_k = [Δt, f(ID), entropy, mean, std]
"""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import pandas as pd

from canlab.config import FEATURES_CFG, PROCESSED_DIR, FEATURES_DIR

logger = logging.getLogger(__name__)


def compute_delta_t(df: pd.DataFrame) -> pd.Series:
    """Calcule le Δt inter-frame en secondes."""
    dt = df["timestamp"].diff().fillna(0.0)
    return dt


def compute_id_frequency(df: pd.DataFrame, window: int | None = None) -> pd.Series:
    """Fréquence de chaque ID dans une fenêtre glissante."""
    if window is None:
        window = FEATURES_CFG.rolling_window
    # Comptage par ID dans la fenêtre
    freq = df.groupby("arb_id")["timestamp"].transform(
        lambda s: s.rolling(window, min_periods=1).count() / (
            s.rolling(window, min_periods=1).apply(lambda x: x.iloc[-1] - x.iloc[0] + 1e-9, raw=False)
        )
    )
    return freq.fillna(0.0)


def compute_payload_entropy(df: pd.DataFrame) -> pd.Series:
    """Entropie de Shannon du payload (sur les bytes)."""
    byte_cols = [c for c in df.columns if c.startswith("byte_")]
    if not byte_cols:
        return pd.Series(0.0, index=df.index)

    def row_entropy(row: pd.Series) -> float:
        vals = row.values.astype(np.float64)
        # Normaliser en distribution
        total = vals.sum()
        if total == 0:
            return 0.0
        probs = vals / total
        probs = probs[probs > 0]
        return -np.sum(probs * np.log2(probs))

    return df[byte_cols].apply(row_entropy, axis=1)


def compute_rolling_stats(
    df: pd.DataFrame, window: int | None = None
) -> pd.DataFrame:
    """Calcule mean et std glissantes sur les bytes du payload."""
    if window is None:
        window = FEATURES_CFG.rolling_window
    byte_cols = [c for c in df.columns if c.startswith("byte_")]
    if not byte_cols:
        return pd.DataFrame(
            {"payload_mean": 0.0, "payload_std": 0.0}, index=df.index
        )

    payload_mean = df[byte_cols].mean(axis=1)
    rolling_mean = payload_mean.rolling(window, min_periods=1).mean()
    rolling_std = payload_mean.rolling(window, min_periods=1).std().fillna(0.0)

    return pd.DataFrame(
        {"payload_rolling_mean": rolling_mean, "payload_rolling_std": rolling_std}
    )


def compute_burstiness(df: pd.DataFrame, window: int | None = None) -> pd.Series:
    """Burstiness : (std_dt - mean_dt) / (std_dt + mean_dt + 1e-9)."""
    if window is None:
        window = FEATURES_CFG.rolling_window
    dt = compute_delta_t(df)
    mean_dt = dt.rolling(window, min_periods=1).mean()
    std_dt = dt.rolling(window, min_periods=1).std().fillna(0.0)
    burstiness = (std_dt - mean_dt) / (std_dt + mean_dt + 1e-9)
    return burstiness.fillna(0.0)


def build_features(df: pd.DataFrame) -> pd.DataFrame:
    """Pipeline complet de feature engineering.

    Retourne un DataFrame avec :
    - timestamp, arb_id (identifiants)
    - delta_t, id_freq, entropy, payload_rolling_mean, payload_rolling_std, burstiness
    """
    logger.info("🔬 Calcul des features sur %d frames...", len(df))

    features = pd.DataFrame()
    features["timestamp"] = df["timestamp"]
    features["arb_id"] = df["arb_id"]

    # Δt inter-frame
    features["delta_t"] = compute_delta_t(df)

    # Fréquence ID
    features["id_freq"] = compute_id_frequency(df)

    # Entropie payload
    features["entropy"] = compute_payload_entropy(df)

    # Rolling stats
    rolling = compute_rolling_stats(df)
    features["payload_rolling_mean"] = rolling["payload_rolling_mean"]
    features["payload_rolling_std"] = rolling["payload_rolling_std"]

    # Burstiness
    features["burstiness"] = compute_burstiness(df)

    # Label (0 = normal, 1 = attaque) — par défaut tout est normal
    features["label"] = 0

    logger.info("✅ Features calculées : shape=%s", features.shape)
    return features


def extract_feature_vector(features_df: pd.DataFrame) -> np.ndarray:
    """Extrait la matrice de features numériques pour le ML.

    Retourne x_k = [delta_t, id_freq, entropy, mean, std, burstiness]
    """
    cols = [
        "delta_t",
        "id_freq",
        "entropy",
        "payload_rolling_mean",
        "payload_rolling_std",
        "burstiness",
    ]
    return features_df[cols].values.astype(np.float32)


# ──────────────────────────────────────────────
# Pipeline complet
# ──────────────────────────────────────────────
def run_pipeline(
    input_path: Path | None = None,
    output_path: Path | None = None,
) -> pd.DataFrame:
    """Exécute le pipeline : charge les données, calcule les features, sauvegarde."""
    if input_path is None:
        input_path = PROCESSED_DIR / "can_frames.parquet"

    if not input_path.exists():
        logger.error("❌ Fichier introuvable : %s", input_path)
        logger.info("💡 Lancez d'abord : python -m canlab.data.ingest")
        raise FileNotFoundError(input_path)

    df = pd.read_parquet(input_path)
    features = build_features(df)

    if output_path is None:
        output_path = FEATURES_DIR / "features.parquet"

    output_path.parent.mkdir(parents=True, exist_ok=True)
    features.to_parquet(output_path, engine="pyarrow")
    logger.info("✅ Features sauvegardées : %s", output_path)
    return features


# ──────────────────────────────────────────────
# CLI
# ──────────────────────────────────────────────
def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description="CAN Feature Engineering")
    parser.add_argument("--input", type=Path, default=None)
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO)
    run_pipeline(input_path=args.input, output_path=args.output)


if __name__ == "__main__":
    main()
