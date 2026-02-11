"""ECU Cluster — Simulation du tableau de bord (affichage vitesse).

Le cluster reçoit les données moteur et affiche la vitesse.
C'est la cible principale de l'attaque stealth :
faire afficher une vitesse différente de la réalité.
"""

from __future__ import annotations

import struct
import logging

import can

from canlab.config import ECU_CLUSTER

logger = logging.getLogger(__name__)


class ClusterECU:
    """Simule l'ECU tableau de bord : affichage vitesse."""

    def __init__(self, bus: can.BusABC | None = None) -> None:
        self.bus = bus
        self.displayed_speed_kmh: float = 0.0
        self.displayed_rpm: int = 0
        self.warning_flags: int = 0  # bitfield pour les alertes

    def update(self, speed_kmh: float, rpm: int = 0) -> None:
        """Met à jour les valeurs affichées."""
        self.displayed_speed_kmh = max(0.0, speed_kmh)
        self.displayed_rpm = max(0, rpm)

    def build_payload(self) -> bytes:
        """Payload : speed (uint16) + RPM (uint16) + warnings (uint8) + padding."""
        speed_raw = int(self.displayed_speed_kmh * 100)
        return struct.pack(
            ">HHBxxx",
            speed_raw,
            self.displayed_rpm,
            self.warning_flags,
        )

    def send_frame(self, speed_kmh: float, rpm: int = 0) -> can.Message | None:
        """Envoie la frame Cluster sur le bus CAN."""
        self.update(speed_kmh, rpm)
        payload = self.build_payload()
        msg = can.Message(
            arbitration_id=ECU_CLUSTER.arb_id,
            data=payload,
            is_extended_id=False,
        )
        if self.bus is not None:
            try:
                self.bus.send(msg)
            except can.CanError as e:
                logger.warning("Cluster ECU send error: %s", e)
                return None
        return msg

    @staticmethod
    def decode_payload(data: bytes) -> dict:
        """Décode un payload Cluster ECU."""
        if len(data) < 5:
            data = data.ljust(8, b"\x00")
        speed_raw, rpm, warnings = struct.unpack(">HHB", data[:5])
        return {
            "displayed_speed_kmh": speed_raw / 100.0,
            "displayed_rpm": rpm,
            "warning_flags": warnings,
        }
