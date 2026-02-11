"""Tests pour les systèmes IDS."""

import numpy as np

from canlab.ids.rules import RuleBasedIDS
from canlab.ids.cusum import CUSUMIDS


class TestRuleBasedIDS:
    """Tests IDS règles."""

    def test_inactive_by_default(self):
        ids = RuleBasedIDS()
        assert not ids.is_active

    def test_activate_deactivate(self):
        ids = RuleBasedIDS()
        ids.activate()
        assert ids.is_active
        ids.deactivate()
        assert not ids.is_active

    def test_no_alerts_when_inactive(self):
        ids = RuleBasedIDS()
        alerts = ids.check_frame(0x100, b"\x03\x20\x00\x64\x00\x00\x1e\x00")
        assert len(alerts) == 0

    def test_valid_frame_no_alert(self):
        ids = RuleBasedIDS()
        ids.activate()
        # Frame Engine normale
        alerts = ids.check_frame(0x100, b"\x03\x20\x00\x64\x1e\x14\x00\x00")
        # Peut avoir des alertes fréquence au début, filtrer
        rule_alerts = [a for a in alerts if a.rule != "frequency"]
        assert len(rule_alerts) == 0

    def test_unknown_id_alert(self):
        ids = RuleBasedIDS()
        ids.activate()
        alerts = ids.check_frame(0x999, b"\x00\x00\x00\x00\x00\x00\x00\x00")
        id_alerts = [a for a in alerts if a.rule == "id_whitelist"]
        assert len(id_alerts) == 1
        assert id_alerts[0].severity == "critical"

    def test_get_summary(self):
        ids = RuleBasedIDS()
        ids.activate()
        summary = ids.get_summary()
        assert "active" in summary
        assert summary["active"] is True


class TestCUSUMIDS:
    """Tests CUSUM IDS."""

    def test_initial_score(self):
        cusum = CUSUMIDS()
        assert cusum.S == 0.0

    def test_calibrate(self):
        cusum = CUSUMIDS()
        X = np.random.default_rng(42).normal(size=(100, 5))
        cusum.calibrate(X)
        assert cusum._calibrated

    def test_normal_traffic_low_score(self):
        cusum = CUSUMIDS()
        rng = np.random.default_rng(42)
        X = rng.normal(size=(100, 5))
        cusum.calibrate(X)
        cusum.activate()

        # Trafic normal → score devrait rester bas
        for _ in range(50):
            x = rng.normal(size=5)
            result = cusum.update(x)
        assert result["cusum_score"] < cusum.threshold

    def test_anomaly_high_score(self):
        cusum = CUSUMIDS(threshold=10.0)
        rng = np.random.default_rng(42)
        X = rng.normal(size=(200, 5))
        cusum.calibrate(X)
        cusum.activate()

        # Injecter des anomalies (shift de 10 sigma)
        for _ in range(100):
            x = rng.normal(size=5) + 10.0  # grand shift
            result = cusum.update(x)

        assert result["cusum_score"] > cusum.threshold
        assert result["is_anomaly"] is True

    def test_reset(self):
        cusum = CUSUMIDS()
        cusum.S = 100.0
        cusum.reset()
        assert cusum.S == 0.0

    def test_get_history(self):
        cusum = CUSUMIDS()
        rng = np.random.default_rng(42)
        X = rng.normal(size=(50, 3))
        cusum.calibrate(X)
        cusum.activate()

        for _ in range(10):
            cusum.update(rng.normal(size=3))

        history = cusum.get_history()
        assert len(history) == 10

    def test_get_summary(self):
        cusum = CUSUMIDS()
        summary = cusum.get_summary()
        assert "active" in summary
        assert "calibrated" in summary
        assert "threshold" in summary
