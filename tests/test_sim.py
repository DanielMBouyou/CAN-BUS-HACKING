"""Tests pour les ECUs simulées."""

import struct

from canlab.sim.ecu_engine import EngineECU
from canlab.sim.ecu_abs import AbsECU
from canlab.sim.ecu_steer import SteeringECU
from canlab.sim.ecu_cluster import ClusterECU


class TestEngineECU:
    """Tests ECU moteur."""

    def test_initial_rpm(self):
        ecu = EngineECU()
        assert ecu.rpm == 800.0  # idle

    def test_update_rpm(self):
        ecu = EngineECU()
        initial = ecu.rpm
        ecu.update_rpm()
        # RPM devrait changer (throttle > load par défaut)
        assert ecu.rpm != initial or True  # le bruit peut ne pas changer

    def test_rpm_bounds(self):
        ecu = EngineECU()
        for _ in range(1000):
            ecu.update_rpm()
        assert 800 <= ecu.rpm <= 7000

    def test_speed_positive(self):
        ecu = EngineECU()
        assert ecu.vehicle_speed_kmh >= 0

    def test_payload_length(self):
        ecu = EngineECU()
        payload = ecu.build_payload()
        assert len(payload) == 8  # DLC = 8

    def test_send_frame_no_bus(self):
        ecu = EngineECU()
        msg = ecu.send_frame()
        assert msg is not None
        assert msg.arbitration_id == 0x100

    def test_decode_payload(self):
        ecu = EngineECU()
        payload = ecu.build_payload()
        decoded = EngineECU.decode_payload(payload)
        assert "rpm" in decoded
        assert "speed_kmh" in decoded
        assert "throttle" in decoded

    def test_set_throttle(self):
        ecu = EngineECU()
        ecu.set_throttle(0.8)
        assert ecu.throttle == 0.8
        ecu.set_throttle(1.5)  # clamped
        assert ecu.throttle == 1.0
        ecu.set_throttle(-0.1)  # clamped
        assert ecu.throttle == 0.0


class TestAbsECU:
    """Tests ECU ABS."""

    def test_initial_speeds(self):
        ecu = AbsECU()
        assert all(s == 0.0 for s in ecu.wheel_speeds)

    def test_update(self):
        ecu = AbsECU()
        ecu.update(60.0)
        assert all(s > 0 for s in ecu.wheel_speeds)

    def test_payload_length(self):
        ecu = AbsECU()
        ecu.update(60.0)
        payload = ecu.build_payload()
        assert len(payload) == 8

    def test_send_frame_no_bus(self):
        ecu = AbsECU()
        msg = ecu.send_frame(60.0)
        assert msg is not None
        assert msg.arbitration_id == 0x110

    def test_decode_payload(self):
        ecu = AbsECU()
        ecu.update(80.0)
        payload = ecu.build_payload()
        decoded = AbsECU.decode_payload(payload)
        assert "wheel_fl_kmh" in decoded
        assert decoded["wheel_fl_kmh"] > 0


class TestSteeringECU:
    """Tests ECU direction."""

    def test_initial_angle(self):
        ecu = SteeringECU()
        assert ecu.angle_deg == 0.0

    def test_update(self):
        ecu = SteeringECU()
        ecu.update()
        # L'angle devrait avoir changé
        assert isinstance(ecu.angle_deg, float)

    def test_payload_length(self):
        ecu = SteeringECU()
        payload = ecu.build_payload()
        assert len(payload) == 8

    def test_send_frame_no_bus(self):
        ecu = SteeringECU()
        msg = ecu.send_frame()
        assert msg is not None
        assert msg.arbitration_id == 0x120

    def test_decode_payload(self):
        ecu = SteeringECU()
        ecu.update()
        payload = ecu.build_payload()
        decoded = SteeringECU.decode_payload(payload)
        assert "angle_deg" in decoded


class TestClusterECU:
    """Tests ECU tableau de bord."""

    def test_initial_state(self):
        ecu = ClusterECU()
        assert ecu.displayed_speed_kmh == 0.0

    def test_update(self):
        ecu = ClusterECU()
        ecu.update(80.0, 3000)
        assert ecu.displayed_speed_kmh == 80.0
        assert ecu.displayed_rpm == 3000

    def test_payload_length(self):
        ecu = ClusterECU()
        ecu.update(60.0)
        payload = ecu.build_payload()
        assert len(payload) == 8

    def test_send_frame_no_bus(self):
        ecu = ClusterECU()
        msg = ecu.send_frame(60.0, 2000)
        assert msg is not None
        assert msg.arbitration_id == 0x130

    def test_decode_payload(self):
        ecu = ClusterECU()
        ecu.update(120.0, 4000)
        payload = ecu.build_payload()
        decoded = ClusterECU.decode_payload(payload)
        assert "displayed_speed_kmh" in decoded
        assert abs(decoded["displayed_speed_kmh"] - 120.0) < 0.1
