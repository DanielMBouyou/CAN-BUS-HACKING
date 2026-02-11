"""Tests pour les modules d'attaque (v2 — optimisé)."""

import numpy as np
import torch

from canlab.attack.mimic_model import (
    CANMimicLSTM,
    CosineWarmupScheduler,
    TemporalAttention,
    prepare_sequences,
    train_mimic_model,
)
from canlab.attack.injector import CANInjector, AttackMode
from canlab.attack.optimizer import create_dummy_scorers, StealthOptimizer


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# LSTM Mimic Model
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━


class TestMimicModel:
    """Tests du modèle LSTM mimic."""

    def test_model_creation(self):
        model = CANMimicLSTM()
        assert model.input_size == 8
        assert model.hidden_size == 128
        assert model.num_layers == 2
        assert model.use_attention is True

    def test_model_no_attention(self):
        model = CANMimicLSTM(use_attention=False)
        assert model.attention is None

    def test_forward_shape(self):
        model = CANMimicLSTM(input_size=8, hidden_size=64, num_layers=1)
        x = torch.randn(4, 50, 8)
        out = model(x)
        assert out.shape == (4, 8)

    def test_forward_with_ecu_ids(self):
        model = CANMimicLSTM(input_size=8, hidden_size=64, num_layers=1)
        x = torch.randn(4, 50, 8)
        ecu_ids = torch.tensor([0, 1, 2, 3])
        out = model(x, ecu_ids)
        assert out.shape == (4, 8)

    def test_forward_without_ecu_ids(self):
        model = CANMimicLSTM(input_size=8, hidden_size=64, num_layers=1)
        x = torch.randn(4, 50, 8)
        out = model(x, ecu_ids=None)
        assert out.shape == (4, 8)

    def test_generate_frame(self):
        model = CANMimicLSTM(input_size=8, hidden_size=64, num_layers=1)
        seq = np.random.rand(50, 8).astype(np.float32)
        frame = model.generate_frame(seq)
        assert frame.shape == (8,)
        assert frame.dtype == np.uint8
        assert all(0 <= b <= 255 for b in frame)

    def test_generate_frame_with_ecu_id(self):
        model = CANMimicLSTM(input_size=8, hidden_size=64, num_layers=1)
        seq = np.random.rand(50, 8).astype(np.float32)
        frame = model.generate_frame(seq, ecu_id=2)
        assert frame.shape == (8,)
        assert frame.dtype == np.uint8

    def test_generate_sequence(self):
        model = CANMimicLSTM(input_size=8, hidden_size=64, num_layers=1)
        seed = np.random.rand(50, 8).astype(np.float32)
        frames = model.generate_sequence(seed, n_steps=5)
        assert len(frames) == 5
        assert all(f.shape == (8,) for f in frames)
        assert all(f.dtype == np.uint8 for f in frames)

    def test_generate_sequence_with_raw_bytes(self):
        """generate_sequence normalise auto si max > 1."""
        model = CANMimicLSTM(input_size=8, hidden_size=64, num_layers=1)
        seed = np.random.randint(0, 256, size=(50, 8)).astype(np.float32)
        frames = model.generate_sequence(seed, n_steps=3)
        assert len(frames) == 3

    def test_output_range(self):
        """Sigmoid → sortie ∈ [0, 1]."""
        model = CANMimicLSTM(input_size=8, hidden_size=64, num_layers=1)
        x = torch.randn(2, 50, 8)
        out = model(x)
        assert (out >= 0).all() and (out <= 1).all()

    def test_prepare_sequences(self):
        payloads = np.random.randint(0, 256, size=(200, 8), dtype=np.uint8)
        X, Y = prepare_sequences(payloads, seq_len=50)
        assert X.shape == (150, 50, 8)
        assert Y.shape == (150, 8)
        assert X.dtype == np.float32
        assert X.max() <= 1.0

    def test_prepare_sequences_with_ids(self):
        payloads = np.random.randint(0, 256, size=(200, 8), dtype=np.uint8)
        ecu_ids = np.random.randint(0, 4, size=200)
        X, Y, ids = prepare_sequences(payloads, seq_len=50, ecu_ids=ecu_ids)
        assert X.shape == (150, 50, 8)
        assert Y.shape == (150, 8)
        assert ids.shape == (150,)


class TestTemporalAttention:
    """Tests du bloc d'attention temporelle."""

    def test_shape(self):
        attn = TemporalAttention(hidden_size=64, n_heads=4)
        x = torch.randn(2, 50, 64)
        out = attn(x)
        assert out.shape == (2, 64)

    def test_single_timestep(self):
        attn = TemporalAttention(hidden_size=64, n_heads=4)
        x = torch.randn(1, 1, 64)
        out = attn(x)
        assert out.shape == (1, 64)


class TestCosineWarmupScheduler:
    """Tests du scheduler LR."""

    def test_warmup_phase(self):
        model = torch.nn.Linear(8, 8)
        optim = torch.optim.Adam(model.parameters(), lr=0.001)
        sched = CosineWarmupScheduler(optim, warmup_epochs=5, total_epochs=50)
        # Après 1 step → LR devrait être < base
        lrs = []
        for _ in range(5):
            lrs.append(sched.get_last_lr()[0])
            sched.step()
        # LR monte pendant le warmup
        assert lrs[-1] >= lrs[0]

    def test_cosine_decay(self):
        model = torch.nn.Linear(8, 8)
        optim = torch.optim.Adam(model.parameters(), lr=0.001)
        sched = CosineWarmupScheduler(optim, warmup_epochs=2, total_epochs=20)
        for _ in range(2):
            sched.step()
        lr_after_warmup = sched.get_last_lr()[0]
        for _ in range(10):
            sched.step()
        lr_mid = sched.get_last_lr()[0]
        assert lr_mid < lr_after_warmup  # decay après warmup


class TestTrainMimicModel:
    """Tests de l'entraînement (mode rapide)."""

    def test_train_returns_history(self):
        payloads = np.random.randint(0, 256, size=(200, 8), dtype=np.uint8)
        model, history = train_mimic_model(payloads, epochs=3, batch_size=16)
        assert isinstance(history, dict)
        assert "train_losses" in history
        assert "val_losses" in history
        assert "best_epoch" in history
        assert "lr_history" in history
        assert len(history["train_losses"]) == 3

    def test_train_model_output_valid(self):
        payloads = np.random.randint(0, 256, size=(200, 8), dtype=np.uint8)
        model, _ = train_mimic_model(payloads, epochs=2, batch_size=16)
        # Vérifier que le modèle produit des sorties valides
        seq = np.random.rand(50, 8).astype(np.float32)
        frame = model.generate_frame(seq)
        assert frame.shape == (8,)


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Injector
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━


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

    def test_jittered_period(self):
        injector = CANInjector()
        periods = [injector._jittered_period(100.0) for _ in range(100)]
        # Toutes les périodes dans ±jitter_pct
        jitter = injector._jitter_pct
        assert all(100.0 * (1 - jitter) <= p <= 100.0 * (1 + jitter) for p in periods)
        # Il y a de la variance
        assert len(set(int(p * 100) for p in periods)) > 1

    def test_smooth_speed(self):
        injector = CANInjector()
        # Premier appel → retourne la valeur directe
        s1 = injector._smooth_speed(100.0)
        assert s1 == 100.0
        # Deuxième appel → moyenne de 100 et 200
        s2 = injector._smooth_speed(200.0)
        assert s2 == 150.0

    def test_ids_feedback_reduces_speed(self):
        injector = CANInjector()

        def high_scorer(frame):
            return 0.9  # IDS alarmé

        injector.set_ids_feedback(high_scorer)
        # Craft une frame dont les 2 premiers bytes encodent une haute vitesse
        payload = bytearray(b"\x4e\x20\x00\x00\x00\x00\x00\x00")  # 200 km/h
        adjusted = injector._apply_ids_feedback(payload)
        # La vitesse devrait être réduite
        adjusted_speed = ((adjusted[0] << 8) | adjusted[1]) / 100.0
        assert adjusted_speed < 200.0


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Optimizer
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━


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
            n_trials=10,
        )
        result = optimizer._random_search()
        assert "best_frame" in result
        assert "ids_score" in result
        assert "speed_delta" in result
        assert "pareto_front" in result
        assert result["best_frame"].shape == (8,)

    def test_random_search_with_base_frame(self):
        ids_scorer, speed_eval = create_dummy_scorers()
        base = np.array([3, 32, 0, 100, 30, 20, 0, 0], dtype=np.uint8)
        optimizer = StealthOptimizer(
            ids_scorer=ids_scorer,
            speed_evaluator=speed_eval,
            n_trials=20,
        )
        result = optimizer._random_search(base_frame=base)
        assert result["best_frame"].shape == (8,)
        # Perturbation locale : chaque byte devrait être proche du base (±30)
        diff = np.abs(result["best_frame"].astype(int) - base.astype(int))
        assert diff.max() <= 30

    def test_temporal_distance(self):
        ids_scorer, speed_eval = create_dummy_scorers()
        recent = [np.array([50, 50, 50, 50, 50, 50, 50, 50], dtype=np.uint8)]
        optimizer = StealthOptimizer(
            ids_scorer=ids_scorer,
            speed_evaluator=speed_eval,
            recent_frames=recent,
        )
        # Frame identique → distance 0
        d_same = optimizer._temporal_distance(recent[0])
        assert d_same == 0.0
        # Frame très différente → distance > 0
        far = np.array([255, 255, 255, 255, 255, 255, 255, 255], dtype=np.uint8)
        d_far = optimizer._temporal_distance(far)
        assert d_far > 0.0

    def test_pareto_dominance(self):
        a = {"ids_score": 0.1, "speed_delta": 50.0}
        b = {"ids_score": 0.5, "speed_delta": 30.0}
        assert StealthOptimizer._dominates(a, b)
        assert not StealthOptimizer._dominates(b, a)
