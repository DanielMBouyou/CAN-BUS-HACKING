"""Bus CAN virtuel — Orchestrateur de la simulation.

Lance les 4 ECUs sur un bus CAN virtuel (vcan0 ou virtual)
et gère le cycle de simulation.
"""

from __future__ import annotations

import asyncio
import logging
import time
from dataclasses import dataclass, field
from typing import Any

import can

from canlab.config import (
    CAN_BUSTYPE,
    CAN_INTERFACE,
    USE_VIRTUAL_BUS,
    ECU_ENGINE,
    ECU_ABS,
    ECU_STEER,
    ECU_CLUSTER,
    LOG_LEVEL,
)
from canlab.sim.ecu_engine import EngineECU
from canlab.sim.ecu_abs import AbsECU
from canlab.sim.ecu_steer import SteeringECU
from canlab.sim.ecu_cluster import ClusterECU

logging.basicConfig(
    level=getattr(logging, LOG_LEVEL, logging.INFO),
    format="%(asctime)s [%(name)s] %(levelname)s — %(message)s",
)
logger = logging.getLogger(__name__)


@dataclass
class CANBusManager:
    """Gestionnaire du bus CAN virtuel et des ECUs."""

    bus: can.BusABC | None = None
    engine: EngineECU = field(default_factory=EngineECU)
    abs_ecu: AbsECU = field(default_factory=AbsECU)
    steering: SteeringECU = field(default_factory=SteeringECU)
    cluster: ClusterECU = field(default_factory=ClusterECU)
    _running: bool = False
    # Compteurs
    frame_count: int = 0
    _listeners: list[Any] = field(default_factory=list)

    def start_bus(self) -> None:
        """Initialise le bus CAN."""
        if USE_VIRTUAL_BUS:
            logger.info("🔧 Mode Virtual Bus (pas de SocketCAN)")
            self.bus = can.Bus(interface="virtual", channel="vcan_sim")
        else:
            logger.info("🔧 Connexion à %s (%s)", CAN_INTERFACE, CAN_BUSTYPE)
            self.bus = can.Bus(interface=CAN_BUSTYPE, channel=CAN_INTERFACE)

        # Injecter le bus dans les ECUs
        self.engine.bus = self.bus
        self.abs_ecu.bus = self.bus
        self.steering.bus = self.bus
        self.cluster.bus = self.bus

    def stop_bus(self) -> None:
        """Arrête le bus CAN."""
        self._running = False
        if self.bus is not None:
            self.bus.shutdown()
            self.bus = None

    def add_listener(self, callback: Any) -> None:
        """Ajoute un listener pour les frames envoyées."""
        self._listeners.append(callback)

    def _notify(self, msg: can.Message | None) -> None:
        if msg is not None:
            for cb in self._listeners:
                try:
                    cb(msg)
                except Exception:
                    pass

    async def run_async(self, duration_s: float = 60.0) -> None:
        """Lance la simulation asynchrone pour une durée donnée."""
        self.start_bus()
        self._running = True
        logger.info("🚗 Simulation CAN démarrée (durée: %.0fs)", duration_s)

        t_start = time.monotonic()
        t_engine = t_start
        t_abs = t_start
        t_steer = t_start
        t_cluster = t_start

        try:
            while self._running:
                now = time.monotonic()
                if now - t_start >= duration_s:
                    break

                # Engine — 10 ms
                if now - t_engine >= ECU_ENGINE.period_ms / 1000.0:
                    msg = self.engine.send_frame()
                    self._notify(msg)
                    self.frame_count += 1
                    t_engine = now

                # ABS — 20 ms
                if now - t_abs >= ECU_ABS.period_ms / 1000.0:
                    msg = self.abs_ecu.send_frame(self.engine.vehicle_speed_kmh)
                    self._notify(msg)
                    self.frame_count += 1
                    t_abs = now

                # Steering — 50 ms
                if now - t_steer >= ECU_STEER.period_ms / 1000.0:
                    msg = self.steering.send_frame()
                    self._notify(msg)
                    self.frame_count += 1
                    t_steer = now

                # Cluster — 100 ms
                if now - t_cluster >= ECU_CLUSTER.period_ms / 1000.0:
                    msg = self.cluster.send_frame(
                        self.engine.vehicle_speed_kmh,
                        int(self.engine.rpm),
                    )
                    self._notify(msg)
                    self.frame_count += 1
                    t_cluster = now

                # Céder le contrôle à l'event loop
                await asyncio.sleep(0.001)

        except KeyboardInterrupt:
            logger.info("⛔ Simulation interrompue par l'utilisateur")
        finally:
            self.stop_bus()
            logger.info(
                "✅ Simulation terminée — %d frames envoyées", self.frame_count
            )

    def run(self, duration_s: float = 60.0) -> None:
        """Lance la simulation (wrapper synchrone)."""
        asyncio.run(self.run_async(duration_s))


# ──────────────────────────────────────────────
# Point d'entrée CLI
# ──────────────────────────────────────────────
def main() -> None:
    """Point d'entrée : python -m canlab.sim.bus"""
    import argparse

    parser = argparse.ArgumentParser(description="CAN Bus Simulator")
    parser.add_argument(
        "--duration", type=float, default=60.0, help="Durée simulation (s)"
    )
    parser.add_argument(
        "--virtual", action="store_true", help="Forcer mode virtual bus"
    )
    args = parser.parse_args()

    if args.virtual:
        import canlab.config as cfg
        cfg.USE_VIRTUAL_BUS = True
        cfg.CAN_BUSTYPE = "virtual"

    manager = CANBusManager()
    manager.run(duration_s=args.duration)


if __name__ == "__main__":
    main()
