"""Optimiseur de furtivité — Optuna / CMA-ES.

Objectif :
    min Score_IDS(x)
    sous contrainte : Speed_target - Speed_real > delta

Utilise Optuna avec le sampler CMA-ES pour optimiser
les perturbations sur les frames CAN générées par le LSTM.
"""

from __future__ import annotations

import logging
from typing import Callable

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

    Méthode black-box : perturbe les bytes du payload
    et évalue le score IDS + l'écart de vitesse.
    """

    def __init__(
        self,
        ids_scorer: Callable[[np.ndarray], float],
        speed_evaluator: Callable[[np.ndarray], float],
        target_speed_delta: float = 30.0,  # km/h d'écart souhaité
        n_bytes: int = 8,
        n_trials: int = 100,
    ) -> None:
        """
        Args:
            ids_scorer: fonction frame → score IDS (0 = indétectable, 1 = détecté)
            speed_evaluator: fonction frame → écart de vitesse produit
            target_speed_delta: écart de vitesse minimum souhaité
            n_bytes: nombre de bytes à optimiser
            n_trials: nombre d'essais Optuna
        """
        self.ids_scorer = ids_scorer
        self.speed_evaluator = speed_evaluator
        self.target_speed_delta = target_speed_delta
        self.n_bytes = n_bytes
        self.n_trials = n_trials

    def _objective(self, trial: "optuna.Trial") -> float:
        """Fonction objectif Optuna.

        Minimise : score_IDS - lambda * max(0, speed_delta - target)
        """
        # Suggérer des perturbations pour chaque byte
        perturbations = np.array(
            [trial.suggest_int(f"byte_{i}", 0, 255) for i in range(self.n_bytes)],
            dtype=np.uint8,
        )

        # Score IDS (à minimiser)
        ids_score = self.ids_scorer(perturbations)

        # Écart de vitesse (à maximiser)
        speed_delta = self.speed_evaluator(perturbations)

        # Pénalité si l'écart de vitesse est insuffisant
        speed_penalty = max(0.0, self.target_speed_delta - speed_delta)

        # Objectif composite
        # On veut : IDS score bas + speed delta élevé
        objective = ids_score + 0.5 * speed_penalty

        return objective

    def optimize(self, base_frame: np.ndarray | None = None) -> dict:
        """Lance l'optimisation et retourne la meilleure perturbation.

        Returns:
            dict avec 'best_frame', 'ids_score', 'speed_delta', 'n_trials'
        """
        if not OPTUNA_AVAILABLE:
            logger.warning("⚠️ Optuna non disponible, utilisation d'une recherche aléatoire")
            return self._random_search(base_frame)

        optuna.logging.set_verbosity(optuna.logging.WARNING)
        study = optuna.create_study(
            direction="minimize",
            sampler=CmaEsSampler(seed=42),
        )
        study.optimize(self._objective, n_trials=self.n_trials, show_progress_bar=False)

        best_params = study.best_params
        best_frame = np.array(
            [best_params[f"byte_{i}"] for i in range(self.n_bytes)],
            dtype=np.uint8,
        )
        ids_score = self.ids_scorer(best_frame)
        speed_delta = self.speed_evaluator(best_frame)

        logger.info(
            "🎯 Optimisation terminée : IDS=%.4f, SpeedΔ=%.1f km/h, Trials=%d",
            ids_score,
            speed_delta,
            self.n_trials,
        )
        return {
            "best_frame": best_frame,
            "ids_score": ids_score,
            "speed_delta": speed_delta,
            "n_trials": self.n_trials,
            "best_value": study.best_value,
        }

    def _random_search(self, base_frame: np.ndarray | None = None) -> dict:
        """Fallback : recherche aléatoire si Optuna non disponible."""
        rng = np.random.default_rng(42)
        best_score = float("inf")
        best_frame = rng.integers(0, 256, size=self.n_bytes, dtype=np.uint8)

        for _ in range(self.n_trials):
            frame = rng.integers(0, 256, size=self.n_bytes, dtype=np.uint8)
            ids_score = self.ids_scorer(frame)
            speed_delta = self.speed_evaluator(frame)
            penalty = max(0.0, self.target_speed_delta - speed_delta)
            score = ids_score + 0.5 * penalty

            if score < best_score:
                best_score = score
                best_frame = frame.copy()

        return {
            "best_frame": best_frame,
            "ids_score": self.ids_scorer(best_frame),
            "speed_delta": self.speed_evaluator(best_frame),
            "n_trials": self.n_trials,
            "best_value": best_score,
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
        # Plus l'entropie est haute, plus c'est suspect
        return min(1.0, entropy / 3.0)

    def speed_evaluator(frame: np.ndarray) -> float:
        """Écart de vitesse simulé."""
        if len(frame) >= 4:
            speed_raw = (frame[2] << 8) | frame[3]
            return abs(speed_raw / 100.0 - 60.0)  # vs 60 km/h nominal
        return 0.0

    return ids_scorer, speed_evaluator
