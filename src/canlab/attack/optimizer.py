"""Optimiseur de furtivité — Optuna / CMA-ES (optimisé v2).

Améliorations v2 :
    1. Multi-objectif Pareto : IDS score + speed delta séparés
    2. LSTM-seeded search : utilise la sortie LSTM comme point initial
    3. Temporal coherence : pénalité si la frame dévie trop des voisines
    4. Perturbation contrôlée autour de base_frame (pas recherche aveugle)
    5. Nombre de trials configurable depuis config

Objectif :
    min  score_IDS(frame)
    max  speed_delta(frame) ≥ target
    min  temporal_distance(frame, voisines)
"""

from __future__ import annotations

import logging
from typing import Callable, Sequence

import numpy as np

try:
    import optuna
    from optuna.samplers import CmaEsSampler

    OPTUNA_AVAILABLE = True
except ImportError:
    OPTUNA_AVAILABLE = False

from canlab.config import ATTACK_CFG

logger = logging.getLogger(__name__)


class StealthOptimizer:
    """Optimise la furtivité des frames CAN attaquantes.

    v2 : perturbe autour d'un base_frame (ex. sortie LSTM)
    au lieu de chercher dans l'espace entier [0,255]^8.
    """

    def __init__(
        self,
        ids_scorer: Callable[[np.ndarray], float],
        speed_evaluator: Callable[[np.ndarray], float],
        target_speed_delta: float = 30.0,
        n_bytes: int = 8,
        n_trials: int | None = None,
        temporal_weight: float | None = None,
        recent_frames: Sequence[np.ndarray] | None = None,
    ) -> None:
        """
        Args:
            ids_scorer: frame → score IDS [0, 1]
            speed_evaluator: frame → écart de vitesse produit
            target_speed_delta: écart minimum souhaité
            n_bytes: nombre de bytes
            n_trials: nombre d'essais (défaut = config)
            temporal_weight: poids de la pénalité temporelle (défaut = config)
            recent_frames: frames récentes pour calcul de cohérence temporelle
        """
        self.ids_scorer = ids_scorer
        self.speed_evaluator = speed_evaluator
        self.target_speed_delta = target_speed_delta
        self.n_bytes = n_bytes
        self.n_trials = n_trials or ATTACK_CFG.optim_n_trials
        self.temporal_weight = (
            temporal_weight if temporal_weight is not None else ATTACK_CFG.optim_temporal_weight
        )
        self.recent_frames = list(recent_frames) if recent_frames else []
        # Historique Pareto front
        self.pareto_front: list[dict] = []

    # ──────────────────────────────────────────────
    # Cohérence temporelle
    # ──────────────────────────────────────────────
    def _temporal_distance(self, frame: np.ndarray) -> float:
        """Distance L2 normalisée entre la frame et les frames récentes."""
        if not self.recent_frames:
            return 0.0
        dists = [
            np.linalg.norm(frame.astype(float) - r.astype(float))
            for r in self.recent_frames[-5:]  # 5 dernières frames
        ]
        # Normaliser par la distance max théorique (sqrt(8 * 255^2) ≈ 721)
        return float(np.mean(dists)) / 721.0

    # ──────────────────────────────────────────────
    # Objectif
    # ──────────────────────────────────────────────
    def _objective(
        self, trial: "optuna.Trial", base_frame: np.ndarray | None = None
    ) -> float:
        """Objectif composite : IDS + speed_penalty + temporal_penalty.

        Si base_frame fourni, perturbe autour de ce point
        (±perturbation_range au lieu de [0,255]).
        """
        if base_frame is not None:
            # Perturbation contrôlée autour du base_frame
            perturb_range = 30  # ±30 bytes max
            frame = np.array(
                [
                    trial.suggest_int(
                        f"byte_{i}",
                        max(0, int(base_frame[i]) - perturb_range),
                        min(255, int(base_frame[i]) + perturb_range),
                    )
                    for i in range(self.n_bytes)
                ],
                dtype=np.uint8,
            )
        else:
            frame = np.array(
                [trial.suggest_int(f"byte_{i}", 0, 255) for i in range(self.n_bytes)],
                dtype=np.uint8,
            )

        ids_score = self.ids_scorer(frame)
        speed_delta = self.speed_evaluator(frame)
        speed_penalty = max(0.0, self.target_speed_delta - speed_delta)
        temporal_penalty = self._temporal_distance(frame)

        objective = (
            ids_score
            + 0.5 * speed_penalty
            + self.temporal_weight * temporal_penalty
        )

        # Stocker dans Pareto (non-dominated solutions)
        trial.set_user_attr("ids_score", ids_score)
        trial.set_user_attr("speed_delta", speed_delta)
        trial.set_user_attr("temporal_dist", temporal_penalty)

        return objective

    # ──────────────────────────────────────────────
    # Pareto dominance
    # ──────────────────────────────────────────────
    @staticmethod
    def _dominates(a: dict, b: dict) -> bool:
        """a domine b si meilleur ou égal sur tous les objectifs et strictement meilleur sur au moins un."""
        return (
            a["ids_score"] <= b["ids_score"]
            and a["speed_delta"] >= b["speed_delta"]
            and (a["ids_score"] < b["ids_score"] or a["speed_delta"] > b["speed_delta"])
        )

    def _update_pareto(self, solution: dict) -> None:
        """Met à jour le front de Pareto."""
        self.pareto_front = [
            p for p in self.pareto_front if not self._dominates(solution, p)
        ]
        if not any(self._dominates(p, solution) for p in self.pareto_front):
            self.pareto_front.append(solution)

    # ──────────────────────────────────────────────
    # Optimisation principale
    # ──────────────────────────────────────────────
    def optimize(self, base_frame: np.ndarray | None = None) -> dict:
        """Lance l'optimisation.

        Args:
            base_frame: frame initiale (ex. sortie LSTM) pour perturbation locale

        Returns:
            dict avec 'best_frame', 'ids_score', 'speed_delta', 'n_trials', 'pareto_front'
        """
        if not OPTUNA_AVAILABLE:
            logger.warning("⚠️ Optuna non disponible → recherche aléatoire")
            return self._random_search(base_frame)

        optuna.logging.set_verbosity(optuna.logging.WARNING)
        study = optuna.create_study(
            direction="minimize",
            sampler=CmaEsSampler(seed=42),
        )

        # Seed avec base_frame si disponible
        if base_frame is not None:
            enqueue_params = {f"byte_{i}": int(base_frame[i]) for i in range(self.n_bytes)}
            study.enqueue_trial(enqueue_params)

        study.optimize(
            lambda trial: self._objective(trial, base_frame),
            n_trials=self.n_trials,
            show_progress_bar=False,
        )

        # Construire le Pareto front
        self.pareto_front = []
        for trial in study.trials:
            if trial.state == optuna.trial.TrialState.COMPLETE:
                sol = {
                    "frame": np.array(
                        [trial.params[f"byte_{i}"] for i in range(self.n_bytes)],
                        dtype=np.uint8,
                    ),
                    "ids_score": trial.user_attrs.get("ids_score", 1.0),
                    "speed_delta": trial.user_attrs.get("speed_delta", 0.0),
                    "value": trial.value,
                }
                self._update_pareto(sol)

        best_params = study.best_params
        best_frame = np.array(
            [best_params[f"byte_{i}"] for i in range(self.n_bytes)],
            dtype=np.uint8,
        )
        ids_score = self.ids_scorer(best_frame)
        speed_delta = self.speed_evaluator(best_frame)

        logger.info(
            "🎯 Optimisation terminée : IDS=%.4f, SpeedΔ=%.1f, Pareto=%d, Trials=%d",
            ids_score, speed_delta, len(self.pareto_front), self.n_trials,
        )
        return {
            "best_frame": best_frame,
            "ids_score": ids_score,
            "speed_delta": speed_delta,
            "n_trials": self.n_trials,
            "best_value": study.best_value,
            "pareto_front": self.pareto_front,
        }

    def _random_search(self, base_frame: np.ndarray | None = None) -> dict:
        """Fallback : recherche aléatoire (améliorée avec perturbation locale)."""
        rng = np.random.default_rng(42)
        best_score = float("inf")
        best_frame = rng.integers(0, 256, size=self.n_bytes, dtype=np.uint8)
        self.pareto_front = []

        for _ in range(self.n_trials):
            if base_frame is not None:
                # Perturbation locale autour du base_frame
                perturbation = rng.integers(-30, 31, size=self.n_bytes)
                frame = np.clip(
                    base_frame.astype(int) + perturbation, 0, 255
                ).astype(np.uint8)
            else:
                frame = rng.integers(0, 256, size=self.n_bytes, dtype=np.uint8)

            ids_score = self.ids_scorer(frame)
            speed_delta = self.speed_evaluator(frame)
            temporal_penalty = self._temporal_distance(frame)
            penalty = max(0.0, self.target_speed_delta - speed_delta)
            score = ids_score + 0.5 * penalty + self.temporal_weight * temporal_penalty

            sol = {
                "frame": frame.copy(),
                "ids_score": ids_score,
                "speed_delta": speed_delta,
                "value": score,
            }
            self._update_pareto(sol)

            if score < best_score:
                best_score = score
                best_frame = frame.copy()

        return {
            "best_frame": best_frame,
            "ids_score": self.ids_scorer(best_frame),
            "speed_delta": self.speed_evaluator(best_frame),
            "n_trials": self.n_trials,
            "best_value": best_score,
            "pareto_front": self.pareto_front,
        }


def create_dummy_scorers() -> tuple[Callable, Callable]:
    """Crée des scorers de démonstration pour les tests."""

    def ids_scorer(frame: np.ndarray) -> float:
        """Score IDS simplifié basé sur l'entropie."""
        vals = frame.astype(np.float64)
        total = vals.sum()
        if total == 0:
            return 1.0
        probs = vals / total
        probs = probs[probs > 0]
        entropy = -np.sum(probs * np.log2(probs))
        return min(1.0, entropy / 3.0)

    def speed_evaluator(frame: np.ndarray) -> float:
        """Écart de vitesse simulé."""
        if len(frame) >= 4:
            speed_raw = (frame[2] << 8) | frame[3]
            return abs(speed_raw / 100.0 - 60.0)
        return 0.0

    return ids_scorer, speed_evaluator
