"""Ingestion des logs CAN bruts.

Parse les fichiers candump (-L format) et les convertit en DataFrame / Parquet.

Format candump -L :
    (timestamp) interface arbitration_id#data

Exemple :
    (1700000000.000000) vcan0 100#0320006400001E00
"""

from __future__ import annotations

import re
import logging
from pathlib import Path
from typing import Generator

import numpy as np
import pandas as pd

from canlab.config import RAW_DIR, PROCESSED_DIR

logger = logging.getLogger(__name__)

# Regex pour parser une ligne candump -L
_CANDUMP_RE = re.compile(
    r"\((\d+\.\d+)\)\s+(\S+)\s+([0-9A-Fa-f]+)#([0-9A-Fa-f]*)"
)


def parse_candump_line(line: str) -> dict | None:
    """Parse une ligne au format candump -L."""
    m = _CANDUMP_RE.match(line.strip())
    if m is None:
        return None
    timestamp = float(m.group(1))
    interface = m.group(2)
    arb_id = int(m.group(3), 16)
    data_hex = m.group(4)
    data_bytes = bytes.fromhex(data_hex) if data_hex else b""
    return {
        "timestamp": timestamp,
        "interface": interface,
        "arb_id": arb_id,
        "dlc": len(data_bytes),
        "data_hex": data_hex,
        "data_bytes": data_bytes,
    }


def iter_candump_file(path: Path) -> Generator[dict, None, None]:
    """Itère sur un fichier candump et yield les frames parsées."""
    with open(path, "r") as f:
        for line_no, line in enumerate(f, 1):
            parsed = parse_candump_line(line)
            if parsed is not None:
                yield parsed
            elif line.strip():
                logger.debug("Ligne %d non parsée: %s", line_no, line.strip()[:80])


def candump_to_dataframe(path: Path) -> pd.DataFrame:
    """Convertit un fichier candump en DataFrame."""
    records = list(iter_candump_file(path))
    if not records:
        logger.warning("Aucune frame parsée dans %s", path)
        return pd.DataFrame()

    df = pd.DataFrame(records)
    # Séparer les bytes du payload en colonnes individuelles
    max_dlc = df["dlc"].max()
    for i in range(max_dlc):
        df[f"byte_{i}"] = df["data_bytes"].apply(
            lambda b, idx=i: b[idx] if idx < len(b) else 0
        )
    df.drop(columns=["data_bytes"], inplace=True)
    df.sort_values("timestamp", inplace=True)
    df.reset_index(drop=True, inplace=True)
    return df


def generate_synthetic_log(
    output_path: Path | None = None,
    duration_s: float = 30.0,
) -> pd.DataFrame:
    """Génère un log CAN synthétique (pour tests sans vcan).

    Simule les 4 ECUs avec leurs fréquences nominales.
    """
    from canlab.config import ALL_ECUS
    from canlab.sim.ecu_engine import EngineECU
    from canlab.sim.ecu_abs import AbsECU
    from canlab.sim.ecu_steer import SteeringECU
    from canlab.sim.ecu_cluster import ClusterECU

    engine = EngineECU()
    abs_ecu = AbsECU()
    steer = SteeringECU()
    cluster = ClusterECU()

    records = []
    t = 0.0
    dt = 0.001  # 1 ms résolution

    while t < duration_s:
        for ecu_def in ALL_ECUS:
            period_s = ecu_def.period_ms / 1000.0
            # Vérifier si c'est le moment d'émettre
            step = int(t / period_s)
            prev_step = int((t - dt) / period_s)
            if step != prev_step or t < dt:
                if ecu_def.arb_id == 0x100:
                    msg = engine.send_frame()
                elif ecu_def.arb_id == 0x110:
                    msg = abs_ecu.send_frame(engine.vehicle_speed_kmh)
                elif ecu_def.arb_id == 0x120:
                    msg = steer.send_frame()
                elif ecu_def.arb_id == 0x130:
                    msg = cluster.send_frame(engine.vehicle_speed_kmh, int(engine.rpm))
                else:
                    continue

                if msg is not None:
                    data_hex = msg.data.hex()
                    record = {
                        "timestamp": 1_700_000_000.0 + t,
                        "interface": "vcan0",
                        "arb_id": msg.arbitration_id,
                        "dlc": len(msg.data),
                        "data_hex": data_hex,
                    }
                    for i in range(len(msg.data)):
                        record[f"byte_{i}"] = msg.data[i]
                    records.append(record)
        t += dt

    df = pd.DataFrame(records)
    df.sort_values("timestamp", inplace=True)
    df.reset_index(drop=True, inplace=True)

    if output_path is not None:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        df.to_parquet(output_path, engine="pyarrow")
        logger.info("✅ Log synthétique sauvegardé : %s (%d frames)", output_path, len(df))

    return df


def ingest(
    input_path: Path | None = None,
    output_path: Path | None = None,
) -> pd.DataFrame:
    """Pipeline d'ingestion principal.

    Si input_path est fourni, parse le fichier candump.
    Sinon, génère un log synthétique.
    """
    if input_path is not None and input_path.exists():
        logger.info("📥 Ingestion de %s", input_path)
        df = candump_to_dataframe(input_path)
    else:
        logger.info("📥 Génération d'un log synthétique (30s)")
        df = generate_synthetic_log(duration_s=30.0)

    if output_path is None:
        output_path = PROCESSED_DIR / "can_frames.parquet"

    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(output_path, engine="pyarrow")
    logger.info("✅ Données sauvegardées : %s (%d frames)", output_path, len(df))
    return df


# ──────────────────────────────────────────────
# CLI
# ──────────────────────────────────────────────
def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description="CAN Data Ingestor")
    parser.add_argument("--input", type=Path, default=None, help="Fichier candump")
    parser.add_argument("--output", type=Path, default=None, help="Fichier parquet sortie")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO)
    ingest(input_path=args.input, output_path=args.output)


if __name__ == "__main__":
    main()
