"""ECU Steering — Simulation de l'angle de direction.

L'angle volant est simulé par un signal sinusoïdal
pour représenter un conduite normale avec corrections.
"""

from __future__ import annotations

import math
import struct
import logging
import time

import can
import numpy as np

from canlab.config import ECU_STEER, PHYSICS

logger = logging.getLogger(__name__)


class SteeringECU:
    """Simule l'ECU direction : angle volant."""

    def __init__(self, bus: can.BusABC | None = None) -> None:
        self.bus = bus
        self.angle_deg: float = 0.0
        self._t0 = time.monotonic()
        self._rng = np.random.default_rng(44)

    def update(self) -> None:
        """Met à jour l'angle volant (sinusoïdal + bruit)."""
        t = time.monotonic() - self._t0
        base = PHYSICS.steer_amplitude_deg * math.sin(
            2 * math.pi * PHYSICS.steer_freq_hz * t
        )
        noise = self._rng.normal(0, 1.0)
        self.angle_deg = base + noise

    def build_payload(self) -> bytes:
        """Payload : angle (int16, centièmes de degré) + rate (int16) + padding."""
        angle_raw = int(self.angle_deg * 100)
        # Rate = dérivée approchée (on utilise 0 pour simplifier)
        rate_raw = 0
        return struct.pack(">hhxxxx", angle_raw, rate_raw)

    def send_frame(self) -> can.Message | None:
        """Envoie la frame Steering sur le bus CAN."""
        self.update()
        payload = self.build_payload()
        msg = can.Message(
            arbitration_id=ECU_STEER.arb_id,
            data=payload,
            is_extended_id=False,
        )
        if self.bus is not None:
            try:
                self.bus.send(msg)
            except can.CanError as e:
                logger.warning("Steering ECU send error: %s", e)
                return None
        return msg

    @staticmethod
    def decode_payload(data: bytes) -> dict:
        """Décode un payload Steering ECU."""
        if len(data) < 4:
            data = data.ljust(8, b"\x00")
        angle_raw, rate_raw = struct.unpack(">hh", data[:4])
        return {
            "angle_deg": angle_raw / 100.0,
            "rate_deg_s": rate_raw / 100.0,
        }
