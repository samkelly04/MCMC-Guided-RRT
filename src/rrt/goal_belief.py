"""Goal belief representation for uncertain goal localization."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Final

import numpy as np


_EPS: Final[float] = 1e-9


@dataclass
class GoalBelief:
    """Gaussian belief over the goal position.

    Maintains a mean and covariance describing where the planner believes the
    goal might be. New sensor observations are fused via a Kalman-style update
    so downstream samplers can use the latest belief when biasing samples.
    """

    mean: np.ndarray
    covariance: np.ndarray
    observation_count: int = 0

    def __post_init__(self) -> None:
        self.mean = np.asarray(self.mean, dtype=float)
        self.covariance = np.asarray(self.covariance, dtype=float)

        if self.mean.ndim != 1:
            raise ValueError("GoalBelief mean must be a 1D vector")
        if self.covariance.shape != (self.dimension, self.dimension):
            raise ValueError(
                "GoalBelief covariance must be square with same dimension as mean"
            )

    @property
    def dimension(self) -> int:
        """Return the dimensionality of the state."""

        return int(self.mean.shape[0])

    def update(
        self, observation: np.ndarray, observation_covariance: np.ndarray
    ) -> None:
        """Fuse a noisy observation of the goal location.

        Args:
            observation: Measured goal location.
            observation_covariance: Covariance of the sensor noise.
        """

        obs = np.asarray(observation, dtype=float)
        obs_cov = np.asarray(observation_covariance, dtype=float)

        if obs.shape != self.mean.shape:
            raise ValueError("Observation has mismatched dimension")
        if obs_cov.shape != self.covariance.shape:
            raise ValueError("Observation covariance has mismatched dimension")

        innovation = obs - self.mean
        innovation_cov = self.covariance + obs_cov
        gain = self.covariance @ np.linalg.inv(innovation_cov)

        self.mean = self.mean + gain @ innovation
        identity = np.eye(self.dimension)
        self.covariance = (identity - gain) @ self.covariance
        self.observation_count += 1

    def expected_distance(self, point: np.ndarray) -> float:
        """Approximate expected distance from a query point to the true goal.

        Uses distance to the belief mean plus a covariance-based penalty that
        encourages conservative behavior when uncertainty is high.
        """

        q = np.asarray(point, dtype=float)
        if q.shape != self.mean.shape:
            raise ValueError("Point has mismatched dimension")

        delta = np.linalg.norm(q - self.mean)
        uncertainty_penalty = np.trace(self.covariance) / max(delta, _EPS)
        return float(delta + 0.5 * uncertainty_penalty)

    def copy(self) -> "GoalBelief":
        """Return a deep copy of the belief."""

        return GoalBelief(
            mean=self.mean.copy(),
            covariance=self.covariance.copy(),
            observation_count=self.observation_count,
        )

