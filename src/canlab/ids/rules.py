"""IDS Règles classiques — Détection par heuristiques.

Vérifie :
1. ID whitelist : seuls les IDs connus sont autorisés
2. Fréquence max : détecte les injections haute fréquence
3. Payload range : détecte les valeurs hors des limites physiques
"""

from __future__ import annotations

import logging
import time
from collections import defaultdict
from dataclasses import dataclass, field

import numpy as np

from canlab.config import VALID_IDS, ECU_BY_ID, IDS_CFG

logger = logging.getLogger(__name__)


@dataclass
class Alert:
    """Alerte IDS."""

    timestamp: float
    rule: str
    arb_id: int
    severity: str  # "low", "medium", "high", "critical"
    message: str
    details: dict = field(default_factory=dict)


class RuleBasedIDS:
    """IDS basé sur des règles déterministes."""

    def __init__(self) -> None:
        self.alerts: list[Alert] = []
        self._frame_times: dict[int, list[float]] = defaultdict(list)
        self._active = False
        # Fenêtre pour le calcul de fréquence
        self._freq_window_s: float = 1.0

        # Payload ranges attendues (par ID)
        self._payload_ranges: dict[int, list[tuple[int, int]]] = {
            0x100: [  # Engine: RPM(0-7000), Speed(0-30000), throttle(0-100), load(0-100)
                (0, 7000 >> 8),  # byte 0: RPM high
                (0, 255),         # byte 1: RPM low
                (0, 300 >> 8),   # byte 2: speed high
                (0, 255),         # byte 3: speed low
                (0, 100),         # byte 4: throttle
                (0, 100),         # byte 5: load
            ],
            0x110: [  # ABS: 4x wheel speed (uint16)
                (0, 300 >> 8), (0, 255),
                (0, 300 >> 8), (0, 255),
                (0, 300 >> 8), (0, 255),
                (0, 300 >> 8), (0, 255),
            ],
            0x120: [  # Steering: angle (int16, peut être négatif)
                (0, 255), (0, 255),  # angle — pas de range check simple pour int16
                (0, 255), (0, 255),  # rate
                (0, 0), (0, 0), (0, 0), (0, 0),  # padding
            ],
            0x130: [  # Cluster: speed, RPM, warnings
                (0, 300 >> 8), (0, 255),  # speed
                (0, 7000 >> 8), (0, 255),  # RPM
                (0, 255),  # warnings
            ],
        }

    def activate(self) -> None:
        """Active l'IDS."""
        self._active = True
        self.alerts.clear()
        logger.info("🛡️ IDS Règles activé")

    def deactivate(self) -> None:
        """Désactive l'IDS."""
        self._active = False
        logger.info("🛡️ IDS Règles désactivé")

    @property
    def is_active(self) -> bool:
        return self._active

    def check_frame(
        self, arb_id: int, data: bytes, timestamp: float | None = None
    ) -> list[Alert]:
        """Vérifie une frame CAN contre toutes les règles.

        Returns:
            Liste d'alertes générées (vide si tout est normal)
        """
        if not self._active:
            return []

        if timestamp is None:
            timestamp = time.time()

        new_alerts: list[Alert] = []

        # 1. ID Whitelist
        alert = self._check_id_whitelist(arb_id, timestamp)
        if alert:
            new_alerts.append(alert)

        # 2. Fréquence
        alert = self._check_frequency(arb_id, timestamp)
        if alert:
            new_alerts.append(alert)

        # 3. Payload range
        alerts = self._check_payload_range(arb_id, data, timestamp)
        new_alerts.extend(alerts)

        self.alerts.extend(new_alerts)
        return new_alerts

    def _check_id_whitelist(self, arb_id: int, timestamp: float) -> Alert | None:
        """Vérifie si l'ID est dans la whitelist."""
        if arb_id not in VALID_IDS:
            return Alert(
                timestamp=timestamp,
                rule="id_whitelist",
                arb_id=arb_id,
                severity="critical",
                message=f"ID 0x{arb_id:03X} inconnu — pas dans la whitelist",
            )
        return None

    def _check_frequency(self, arb_id: int, timestamp: float) -> Alert | None:
        """Vérifie si la fréquence d'un ID est anormalement élevée."""
        self._frame_times[arb_id].append(timestamp)

        # Nettoyer les anciens timestamps
        cutoff = timestamp - self._freq_window_s
        self._frame_times[arb_id] = [
            t for t in self._frame_times[arb_id] if t > cutoff
        ]

        count = len(self._frame_times[arb_id])
        ecu_def = ECU_BY_ID.get(arb_id)
        if ecu_def is None:
            return None

        expected_freq = 1000.0 / ecu_def.period_ms  # frames/s
        max_freq = expected_freq * IDS_CFG.max_freq_tolerance

        if count > max_freq:
            return Alert(
                timestamp=timestamp,
                rule="frequency",
                arb_id=arb_id,
                severity="high",
                message=(
                    f"ID 0x{arb_id:03X} fréquence anormale: "
                    f"{count:.0f} fps (max={max_freq:.0f})"
                ),
                details={"observed": count, "max": max_freq},
            )
        return None

    def _check_payload_range(
        self, arb_id: int, data: bytes, timestamp: float
    ) -> list[Alert]:
        """Vérifie si les bytes du payload sont dans les ranges attendues."""
        ranges = self._payload_ranges.get(arb_id)
        if ranges is None:
            return []

        alerts = []
        for i, (lo, hi) in enumerate(ranges):
            if i >= len(data):
                break
            if data[i] < lo or data[i] > hi:
                alerts.append(
                    Alert(
                        timestamp=timestamp,
                        rule="payload_range",
                        arb_id=arb_id,
                        severity="medium",
                        message=(
                            f"ID 0x{arb_id:03X} byte[{i}]={data[i]} "
                            f"hors range [{lo}, {hi}]"
                        ),
                        details={"byte_index": i, "value": data[i], "range": (lo, hi)},
                    )
                )
        return alerts

    def get_score(self) -> float:
        """Score IDS global : proportion de frames alertées dans la dernière seconde."""
        if not self.alerts:
            return 0.0
        now = time.time()
        recent = [a for a in self.alerts if now - a.timestamp < 1.0]
        return min(1.0, len(recent) / 10.0)

    def get_summary(self) -> dict:
        """Résumé de l'état de l'IDS."""
        return {
            "active": self._active,
            "total_alerts": len(self.alerts),
            "alerts_by_rule": self._count_by_rule(),
            "alerts_by_severity": self._count_by_severity(),
            "current_score": self.get_score(),
        }

    def _count_by_rule(self) -> dict[str, int]:
        counts: dict[str, int] = defaultdict(int)
        for a in self.alerts:
            counts[a.rule] += 1
        return dict(counts)

    def _count_by_severity(self) -> dict[str, int]:
        counts: dict[str, int] = defaultdict(int)
        for a in self.alerts:
            counts[a.severity] += 1
        return dict(counts)
