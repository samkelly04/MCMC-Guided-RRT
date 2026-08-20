"""Goal belief representation for uncertain goal localization."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Final

import numpy as np

from .goal_sensor import GoalSensorConfig


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
        """Compute distance from a query point to the belief mean.

        Returns the Euclidean distance ||q - μ|| to the belief mean, which serves
        as the exploitation term in the composite cost function:

            J(q) = ||q - μ|| - λ_info · I(q, Σ)

        This formulation provides a principled separation of concerns:
        - Exploitation (this term): Convex function with global minimum at μ,
          driving the agent toward the most likely goal location.
        - Exploration (information gain term): Rewards locations that reduce
          uncertainty, controlled explicitly via λ_info.

        The exploration-exploitation tradeoff is managed through the information
        gain term, not through artificial repulsion or uncertainty penalties in
        the distance computation. This avoids the "volcano effect" of alternative
        approximations (e.g., far-field Taylor expansion) which incorrectly place
        the cost minimum at a ring around μ rather than at μ itself.

        As observations accumulate and Σ decreases, the achievable information gain
        diminishes, causing the cost function to converge naturally to this pure
        distance-to-goal metric.
        """
        q = np.asarray(point, dtype=float)
        if q.shape != self.mean.shape:
            raise ValueError("Point has mismatched dimension")

        return float(np.linalg.norm(q - self.mean))

    def predict_information_gain(
        self, query_point: np.ndarray, sensor_config: GoalSensorConfig | dict
    ) -> float:
        """Predict information gain (entropy reduction) from observing at query_point.

        Computes how much the belief uncertainty would decrease if a measurement
        were taken from the given query point. This is a hypothetical calculation
        that does NOT modify the actual belief state.

        Args:
            query_point: Hypothetical robot position where measurement would be taken.
            sensor_config: Either GoalSensorConfig object or dict with keys
                {'far_std', 'near_std', 'distance_scale'} for noise model.

        Returns:
            Information gain as reduction in trace of covariance: trace(Σ) - trace(Σ_new).
        """
        q = np.asarray(query_point, dtype=float)
        if q.shape != self.mean.shape:
            raise ValueError("Query point has mismatched dimension")

        # Extract sensor config parameters (support both dataclass and dict)
        if isinstance(sensor_config, dict):
            far_std = sensor_config.get("far_std", 8.0)
            near_std = sensor_config.get("near_std", 0.5)
            distance_scale = sensor_config.get("distance_scale", 50.0)
        else:
            far_std = sensor_config.far_std
            near_std = sensor_config.near_std
            distance_scale = sensor_config.distance_scale

        # Compute distance-dependent observation noise (same logic as GoalObservationModel)
        dist = np.linalg.norm(q - self.mean)
        alpha = min(dist / max(distance_scale, 1e-6), 1.0)
        sigma = near_std + (far_std - near_std) * alpha

        # Observation noise covariance (isotropic)
        R = np.eye(self.dimension) * (sigma**2)

        # Hypothetical Kalman update (without modifying self.mean or self.covariance)
        innovation_cov = self.covariance + R  # S = Σ + R
        kalman_gain = self.covariance @ np.linalg.inv(innovation_cov)  # K = Σ · S⁻¹
        identity = np.eye(self.dimension)
        posterior_cov = (identity - kalman_gain) @ self.covariance  # Σ_new = (I - K) · Σ

        # Information gain = reduction in uncertainty (trace of covariance)
        info_gain = np.trace(self.covariance) - np.trace(posterior_cov)
        return float(max(info_gain, 0.0))  # Clamp to non-negative

    def compute_entropy(self) -> float:
        """Compute entropy proxy using trace of covariance matrix.

        For a Gaussian distribution N(μ, Σ), entropy is proportional to
        log(det(Σ)). We use trace(Σ) as a simpler proxy that captures
        the total uncertainty across all dimensions.

        Returns:
            Entropy proxy: trace(covariance)
        """
        return float(np.trace(self.covariance))

    def sample(self, num_samples: int = 1) -> np.ndarray:
        """Sample from the goal belief distribution.

        Args:
            num_samples: Number of samples to draw from the Gaussian distribution.

        Returns:
            Array of shape (num_samples, dimension) containing samples from N(mean, covariance).
        """
        samples = np.random.multivariate_normal(
            mean=self.mean,
            cov=self.covariance,
            size=num_samples
        )
        return samples

    def copy(self) -> "GoalBelief":
        """Return a deep copy of the belief."""

        return GoalBelief(
            mean=self.mean.copy(),
            covariance=self.covariance.copy(),
            observation_count=self.observation_count,
        )

