"""ECU Engine — Simulation du moteur (RPM).

Modèle dynamique simplifié :
    RPM_{k+1} = RPM_k + alpha * (throttle - load) + noise
    Vitesse = (2 * pi * R / 60) * RPM
"""

from __future__ import annotations

import math
import struct
import time
import logging
from typing import TYPE_CHECKING

import can
import numpy as np

from canlab.config import ECU_ENGINE, PHYSICS

if TYPE_CHECKING:
    from canlab.sim.bus import CANBusManager

logger = logging.getLogger(__name__)


class EngineECU:
    """Simule l'ECU moteur qui émet les RPM sur le bus CAN."""

    def __init__(self, bus: can.BusABC | None = None) -> None:
        self.bus = bus
        self.rpm: float = 800.0  # idle
        self.throttle: float = PHYSICS.throttle_default
        self.load: float = PHYSICS.load_default
        self._running = False
        self._rng = np.random.default_rng(42)

    @property
    def vehicle_speed_kmh(self) -> float:
        """Vitesse véhicule dérivée du RPM (km/h)."""
        # v = (2*pi*R / 60) * RPM  → m/s → km/h
        v_ms = (2 * math.pi * PHYSICS.wheel_radius_m / 60.0) * self.rpm
        return v_ms * 3.6

    def update_rpm(self) -> None:
        """Mise à jour du RPM avec le modèle dynamique."""
        noise = self._rng.normal(0, PHYSICS.noise_std)
        self.rpm += PHYSICS.alpha * (self.throttle - self.load) * 100 + noise
        self.rpm = max(PHYSICS.rpm_min, min(PHYSICS.rpm_max, self.rpm))

    def build_payload(self) -> bytes:
        """Construit le payload CAN : RPM (uint16) + speed (uint16) + padding."""
        rpm_int = int(self.rpm)
        speed_int = int(self.vehicle_speed_kmh * 100)  # centièmes de km/h
        # Payload : [RPM_H, RPM_L, SPD_H, SPD_L, throttle%, load%, 0, 0]
        return struct.pack(
            ">HHBBBB",
            rpm_int,
            speed_int,
            int(self.throttle * 100),
            int(self.load * 100),
            0,
            0,
        )

    def send_frame(self) -> can.Message | None:
        """Envoie une frame CAN avec les données moteur."""
        self.update_rpm()
        payload = self.build_payload()
        msg = can.Message(
            arbitration_id=ECU_ENGINE.arb_id,
            data=payload,
            is_extended_id=False,
        )
        if self.bus is not None:
            try:
                self.bus.send(msg)
            except can.CanError as e:
                logger.warning("Engine ECU send error: %s", e)
                return None
        return msg

    def set_throttle(self, value: float) -> None:
        """Modifie le throttle (0.0 — 1.0)."""
        self.throttle = max(0.0, min(1.0, value))

    def set_load(self, value: float) -> None:
        """Modifie la charge (0.0 — 1.0)."""
        self.load = max(0.0, min(1.0, value))

    @staticmethod
    def decode_payload(data: bytes) -> dict:
        """Décode un payload Engine ECU."""
        if len(data) < 6:
            return {}
        rpm, speed_raw, throttle, load, _, _ = struct.unpack(">HHBBBB", data[:8].ljust(8, b"\x00"))
        return {
            "rpm": rpm,
            "speed_kmh": speed_raw / 100.0,
            "throttle": throttle / 100.0,
            "load": load / 100.0,
        }
