"""Tests pour le pipeline de données."""

import numpy as np
import pandas as pd

from canlab.data.ingest import parse_candump_line, candump_to_dataframe
from canlab.data.features import (
    compute_delta_t,
    compute_payload_entropy,
    compute_rolling_stats,
    compute_burstiness,
    build_features,
    extract_feature_vector,
)


class TestIngest:
    """Tests pour l'ingestion des données."""

    def test_parse_valid_line(self):
        line = "(1700000000.123456) vcan0 100#0320006400001E00"
        result = parse_candump_line(line)
        assert result is not None
        assert result["arb_id"] == 0x100
        assert result["dlc"] == 8
        assert abs(result["timestamp"] - 1700000000.123456) < 1e-6

    def test_parse_invalid_line(self):
        result = parse_candump_line("not a valid line")
        assert result is None

    def test_parse_empty_payload(self):
        line = "(1700000000.000000) vcan0 100#"
        result = parse_candump_line(line)
        assert result is not None
        assert result["dlc"] == 0

    def test_parse_short_payload(self):
        line = "(1700000000.000000) vcan0 100#0320"
        result = parse_candump_line(line)
        assert result is not None
        assert result["dlc"] == 2


class TestFeatures:
    """Tests pour le feature engineering."""

    def _make_sample_df(self, n: int = 100) -> pd.DataFrame:
        """Crée un DataFrame de test."""
        rng = np.random.default_rng(42)
        data = {
            "timestamp": np.linspace(1700000000, 1700000001, n),
            "arb_id": rng.choice([0x100, 0x110, 0x120, 0x130], size=n),
            "dlc": np.full(n, 8),
            "data_hex": ["0320006400001E00"] * n,
        }
        for i in range(8):
            data[f"byte_{i}"] = rng.integers(0, 256, size=n)
        return pd.DataFrame(data)

    def test_compute_delta_t(self):
        df = self._make_sample_df()
        dt = compute_delta_t(df)
        assert len(dt) == len(df)
        assert dt.iloc[0] == 0.0  # premier = 0
        assert all(dt.iloc[1:] >= 0)  # tous positifs

    def test_compute_payload_entropy(self):
        df = self._make_sample_df()
        entropy = compute_payload_entropy(df)
        assert len(entropy) == len(df)
        assert all(entropy >= 0)

    def test_compute_rolling_stats(self):
        df = self._make_sample_df()
        stats = compute_rolling_stats(df)
        assert "payload_rolling_mean" in stats.columns
        assert "payload_rolling_std" in stats.columns
        assert len(stats) == len(df)

    def test_compute_burstiness(self):
        df = self._make_sample_df()
        burst = compute_burstiness(df)
        assert len(burst) == len(df)

    def test_build_features(self):
        df = self._make_sample_df()
        features = build_features(df)
        assert "delta_t" in features.columns
        assert "entropy" in features.columns
        assert "burstiness" in features.columns
        assert "label" in features.columns
        assert len(features) == len(df)

    def test_extract_feature_vector(self):
        df = self._make_sample_df()
        features = build_features(df)
        X = extract_feature_vector(features)
        assert X.shape[0] == len(df)
        assert X.shape[1] == 6  # 6 features
        assert X.dtype == np.float32
