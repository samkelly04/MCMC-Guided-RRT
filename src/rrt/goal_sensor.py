"""Simple goal observation sensor model with distance-dependent noise."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

import numpy as np


@dataclass(frozen=True)
class GoalSensorConfig:
    """Configuration for the goal observation model.

    Attributes:
        far_std: Standard deviation applied when the robot is far from the goal.
        near_std: Standard deviation when the robot is very close to the goal.
        distance_scale: Distance at which the noise transitions from far_std
            toward near_std (controls how quickly confidence improves).
        seed: Optional random seed for reproducible observations.
    """

    far_std: float = 8.0
    near_std: float = 0.5
    distance_scale: float = 50.0
    seed: int | None = None


class GoalObservationModel:
    """Generates noisy observations of the true goal location.

    The observation noise follows an isotropic Gaussian distribution whose
    standard deviation shrinks as the robot approaches the goal, mimicking the
    behavior of a sensor that becomes more accurate with proximity.
    """

    def __init__(self, config: GoalSensorConfig):
        self._config = config
        self._rng = np.random.default_rng(config.seed)

    def observe(
        self, robot_position: np.ndarray, true_goal: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Return a noisy observation of the goal and its covariance matrix.

        Args:
            robot_position: Current robot position [x, y].
            true_goal: Ground-truth goal position [x, y].

        Returns:
            observation: Noisy goal measurement.
            covariance: Corresponding isotropic covariance matrix.
        """

        robot_pos = np.asarray(robot_position, dtype=float)
        goal = np.asarray(true_goal, dtype=float)

        if robot_pos.shape != goal.shape:
            raise ValueError("Robot position and goal must share the same dimension")

        distance = np.linalg.norm(goal - robot_pos)
        sigma = self._noise_std(distance)

        observation = goal + self._rng.normal(0.0, sigma, size=goal.shape)
        covariance = np.eye(goal.shape[0]) * (sigma**2)
        return observation, covariance

    def _noise_std(self, distance: float) -> float:
        """Compute the noise standard deviation for a given distance."""

        config = self._config
        alpha = min(distance / max(config.distance_scale, 1e-6), 1.0)
        # Interpolate between far_std (alpha=1) and near_std (alpha=0)
        return config.near_std + (config.far_std - config.near_std) * alpha

    def reset(self, seed: int | None = None) -> None:
        """Reset the random generator for reproducible observation sequences."""

        actual_seed = seed if seed is not None else self._config.seed
        self._rng = np.random.default_rng(actual_seed)

