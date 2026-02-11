"""Tests pour les modules d'attaque."""

import numpy as np
import torch

from canlab.attack.mimic_model import CANMimicLSTM, prepare_sequences
from canlab.attack.injector import CANInjector, AttackMode
from canlab.attack.optimizer import create_dummy_scorers, StealthOptimizer


class TestMimicModel:
    """Tests du modèle LSTM mimic."""

    def test_model_creation(self):
        model = CANMimicLSTM()
        assert model.input_size == 8
        assert model.hidden_size == 128
        assert model.num_layers == 2

    def test_forward_shape(self):
        model = CANMimicLSTM(input_size=8, hidden_size=64, num_layers=1)
        x = torch.randn(4, 50, 8)  # batch=4, seq=50, features=8
        out = model(x)
        assert out.shape == (4, 8)

    def test_generate_frame(self):
        model = CANMimicLSTM(input_size=8, hidden_size=64, num_layers=1)
        seq = np.random.rand(50, 8).astype(np.float32)
        frame = model.generate_frame(seq)
        assert frame.shape == (8,)
        assert frame.dtype == np.uint8
        assert all(0 <= b <= 255 for b in frame)

    def test_prepare_sequences(self):
        payloads = np.random.randint(0, 256, size=(200, 8), dtype=np.uint8)
        X, Y = prepare_sequences(payloads, seq_len=50)
        assert X.shape == (150, 50, 8)
        assert Y.shape == (150, 8)
        assert X.dtype == np.float32
        assert X.max() <= 1.0  # normalisé


class TestInjector:
    """Tests de l'injecteur."""

    def test_craft_naive_frame(self):
        injector = CANInjector()
        msg = injector.craft_naive_frame()
        assert msg.arbitration_id == 0x130
        assert len(msg.data) == 8

    def test_craft_stealth_frame(self):
        injector = CANInjector()
        msg = injector.craft_stealth_frame()
        assert msg.arbitration_id == 0x130
        assert len(msg.data) >= 6

    def test_inject_no_bus(self):
        injector = CANInjector()
        msg = injector.inject_frame()
        assert msg is not None
        assert injector._frame_count == 1

    def test_observe_frame(self):
        import can

        injector = CANInjector()
        msg = can.Message(
            arbitration_id=0x100,
            data=b"\x03\x20\x00\x64\x1e\x14\x00\x00",
        )
        injector.observe_frame(msg)
        assert len(injector._frame_buffer) == 1

    def test_attack_modes(self):
        assert AttackMode.NAIVE == "naive"
        assert AttackMode.STEALTH == "stealth"


class TestOptimizer:
    """Tests de l'optimiseur de furtivité."""

    def test_dummy_scorers(self):
        ids_scorer, speed_eval = create_dummy_scorers()
        frame = np.array([3, 32, 0, 100, 30, 20, 0, 0], dtype=np.uint8)
        score = ids_scorer(frame)
        assert 0 <= score <= 1.0
        delta = speed_eval(frame)
        assert isinstance(delta, float)

    def test_random_search(self):
        ids_scorer, speed_eval = create_dummy_scorers()
        optimizer = StealthOptimizer(
            ids_scorer=ids_scorer,
            speed_evaluator=speed_eval,
            n_trials=10,  # peu de trials pour le test
        )
        result = optimizer._random_search()
        assert "best_frame" in result
        assert "ids_score" in result
        assert "speed_delta" in result
        assert result["best_frame"].shape == (8,)
