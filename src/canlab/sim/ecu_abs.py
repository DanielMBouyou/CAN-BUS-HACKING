"""ECU ABS — Simulation du système antiblocage (vitesse roues).

Chaque roue a une vitesse dérivée de la vitesse véhicule avec du bruit
pour simuler les variations de grip.
"""

from __future__ import annotations

import struct
import logging

import can
import numpy as np

from canlab.config import ECU_ABS

logger = logging.getLogger(__name__)


class AbsECU:
    """Simule l'ECU ABS : vitesse des 4 roues."""

    def __init__(self, bus: can.BusABC | None = None) -> None:
        self.bus = bus
        self._rng = np.random.default_rng(43)
        # Vitesses roues en km/h
        self.wheel_speeds: list[float] = [0.0, 0.0, 0.0, 0.0]

    def update(self, vehicle_speed_kmh: float) -> None:
        """Met à jour les vitesses de roues à partir de la vitesse véhicule."""
        for i in range(4):
            noise = self._rng.normal(0, 0.5)
            self.wheel_speeds[i] = max(0.0, vehicle_speed_kmh + noise)

    def build_payload(self) -> bytes:
        """Payload : 4 x uint16 vitesse roue (centièmes de km/h)."""
        values = [int(s * 100) for s in self.wheel_speeds]
        return struct.pack(">HHHH", *values)

    def send_frame(self, vehicle_speed_kmh: float) -> can.Message | None:
        """Envoie la frame ABS sur le bus CAN."""
        self.update(vehicle_speed_kmh)
        payload = self.build_payload()
        msg = can.Message(
            arbitration_id=ECU_ABS.arb_id,
            data=payload,
            is_extended_id=False,
        )
        if self.bus is not None:
            try:
                self.bus.send(msg)
            except can.CanError as e:
                logger.warning("ABS ECU send error: %s", e)
                return None
        return msg

    @staticmethod
    def decode_payload(data: bytes) -> dict:
        """Décode un payload ABS ECU."""
        if len(data) < 8:
            data = data.ljust(8, b"\x00")
        fl, fr, rl, rr = struct.unpack(">HHHH", data[:8])
        return {
            "wheel_fl_kmh": fl / 100.0,
            "wheel_fr_kmh": fr / 100.0,
            "wheel_rl_kmh": rl / 100.0,
            "wheel_rr_kmh": rr / 100.0,
        }
