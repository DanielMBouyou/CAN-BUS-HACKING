"""Tests pour la configuration."""

from canlab.config import (
    ALL_ECUS,
    ECU_BY_ID,
    ECU_ENGINE,
    ECU_ABS,
    ECU_STEER,
    ECU_CLUSTER,
    PHYSICS,
    VALID_IDS,
)


def test_ecu_definitions():
    """Vérifie que les 4 ECUs sont bien définies."""
    assert len(ALL_ECUS) == 4
    assert ECU_ENGINE.arb_id == 0x100
    assert ECU_ABS.arb_id == 0x110
    assert ECU_STEER.arb_id == 0x120
    assert ECU_CLUSTER.arb_id == 0x130


def test_ecu_by_id():
    """Vérifie le mapping ID → ECU."""
    assert ECU_BY_ID[0x100] == ECU_ENGINE
    assert ECU_BY_ID[0x110] == ECU_ABS
    assert ECU_BY_ID[0x120] == ECU_STEER
    assert ECU_BY_ID[0x130] == ECU_CLUSTER


def test_valid_ids():
    """Vérifie l'ensemble des IDs valides."""
    assert VALID_IDS == {0x100, 0x110, 0x120, 0x130}


def test_physics_params():
    """Vérifie les paramètres physiques."""
    assert PHYSICS.rpm_min < PHYSICS.rpm_max
    assert 0 < PHYSICS.wheel_radius_m < 1.0
    assert PHYSICS.alpha > 0


def test_ecu_periods():
    """Vérifie les fréquences d'émission."""
    assert ECU_ENGINE.period_ms == 10
    assert ECU_ABS.period_ms == 20
    assert ECU_STEER.period_ms == 50
    assert ECU_CLUSTER.period_ms == 100
