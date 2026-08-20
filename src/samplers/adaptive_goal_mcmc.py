"""MCMC sampler using Metropolis-within-Gibbs for goal-directed RRT planning."""

from __future__ import annotations

from typing import Final
import numpy as np
from shapely.geometry import Point, Polygon

from .base import BaseSampler
from ..rrt.goal_belief import GoalBelief
from ..rrt.collision import point_is_free


_EPS: Final[float] = 1e-9
_INFINITE_COST: Final[float] = 1e10


class AdaptiveGoalMCMCSampler(BaseSampler):
    """MCMC sampler targeting a Boltzmann distribution for goal-directed planning.

    Uses Metropolis-within-Gibbs to generate samples from π(q) ∝ exp(-J(q) / T),
    where J(q) combines goal proximity and obstacle clearance. Maintains chain
    state across samples for efficient exploration.
    """

    def __init__(
        self,
        proposal_std: float = 5.0,
        temperature: float = 10.0,
        lambda_goal: float = 1.0,
        lambda_clearance: float = 0.5,
        clearance_epsilon: float = 0.1,
        uniform_mixing_rate: float = 0.1,
        lambda_info: float = 2.0,
        seed: int | None = None,
    ):
        """Initialize the MCMC sampler.

        Args:
            proposal_std: Standard deviation of Gaussian proposal distribution.
            temperature: Temperature T in Boltzmann distribution (higher = more exploration).
            lambda_goal: Weight for goal proximity term in cost function.
            lambda_clearance: Weight for obstacle clearance term in cost function.
            clearance_epsilon: Small value to prevent division by zero in clearance.
            uniform_mixing_rate: Probability λ of selecting uniform sample vs MCMC (default 0.1).
            lambda_info: Weight for information gain term in cost function (default 2.0).
            seed: Optional random seed for reproducibility.
        """
        self.proposal_std = float(proposal_std)
        self.temperature = float(temperature)
        self.lambda_goal = float(lambda_goal)
        self.lambda_clearance = float(lambda_clearance)
        self.clearance_epsilon = float(clearance_epsilon)
        self.uniform_mixing_rate = float(uniform_mixing_rate)
        self.lambda_info = float(lambda_info)

        self.rng = np.random.default_rng(seed)
        self._current_state: np.ndarray | None = None
        self._dimension: int | None = None
        self.sample_log: list = []

    def sample(
        self, bounds: np.ndarray, state: dict | None = None
    ) -> np.ndarray:
        """Generate a sample using hybrid λ-biased sampling strategy.

        With probability uniform_mixing_rate (λ), returns a uniform random sample.
        Otherwise, performs MCMC Metropolis-within-Gibbs step. The MCMC chain state
        persists across iterations, pausing when uniform samples are selected.

        Args:
            bounds: Workspace limits with shape (2, D): [[min...], [max...]]
            state: Optional state dict containing 'goal_belief' and 'obstacles'

        Returns:
            Sample point as numpy array with shape (D,)
        """
        bounds = np.asarray(bounds, dtype=float)

        goal_belief: GoalBelief | None = None
        obstacles: list[Polygon] | None = None
        sensor_config = None

        if state is not None:
            goal_belief = state.get("goal_belief")
            obstacles = state.get("obstacles")
            sensor_config = state.get("sensor_config")

        if self._dimension is None:
            self._dimension = len(bounds[0])

        # Lambda-biasing: probabilistically choose between uniform and MCMC sampling
        r = self.rng.random()
        if r < self.uniform_mixing_rate:
            # Uniform exploration mode: generate collision-free uniform sample
            # MCMC chain state persists (is NOT reset) to resume on next MCMC iteration
            q_uniform = self._sample_uniform(bounds, obstacles)
            self.sample_log.append((q_uniform.copy(), "uniform", 0.0))
            return q_uniform

        # MCMC exploitation mode: perform Metropolis-within-Gibbs step
        if self._current_state is None:
            self._current_state = self._sample_uniform_initial(bounds, obstacles)
            self.sample_log.append((self._current_state.copy(), "accepted", 0.0))
            return self._current_state.copy()

        q_proposed = self._propose_new_state(bounds)

        # Compute cost to classify rejection type before accept/reject decision
        cost_proposed = self._compute_cost(
            q_proposed, goal_belief, obstacles, bounds, sensor_config
        )

        if cost_proposed >= _INFINITE_COST:
            # Collision rejection: proposal hit obstacle or out of bounds
            self.sample_log.append((q_proposed.copy(), "collision", cost_proposed))
        else:
            # Collision-free proposal: compute acceptance via Metropolis rule
            cost_current = self._compute_cost(
                self._current_state, goal_belief, obstacles, bounds, sensor_config
            )
            log_acceptance = min(
                0.0, -cost_proposed / self.temperature + cost_current / self.temperature
            )

            if self._accept_proposal(log_acceptance):
                self._current_state = q_proposed
                self.sample_log.append((q_proposed.copy(), "accepted", cost_proposed))
            else:
                self.sample_log.append(
                    (q_proposed.copy(), "metropolis_rejected", cost_proposed)
                )

        return self._current_state.copy()

    def _sample_uniform_initial(
        self, bounds: np.ndarray, obstacles: list[Polygon] | None
    ) -> np.ndarray:
        """Sample an initial point uniformly, ensuring it's collision-free."""
        max_attempts = 100
        for _ in range(max_attempts):
            q = self.rng.uniform(bounds[0], bounds[1])
            if obstacles is None or point_is_free(q, obstacles):
                return q

        return (bounds[0] + bounds[1]) / 2.0

    def _sample_uniform(
        self, bounds: np.ndarray, obstacles: list[Polygon] | None
    ) -> np.ndarray:
        """Sample a uniform random point in the workspace, ensuring it's collision-free.

        Args:
            bounds: Workspace limits with shape (2, D): [[min...], [max...]]
            obstacles: Optional list of obstacle polygons for collision checking

        Returns:
            Collision-free uniform random sample with shape (D,)
        """
        max_attempts = 100
        for _ in range(max_attempts):
            q = self.rng.uniform(bounds[0], bounds[1])
            if obstacles is None or point_is_free(q, obstacles):
                return q

        # Fallback to workspace center if all attempts collide
        return (bounds[0] + bounds[1]) / 2.0

    def _propose_new_state(self, bounds: np.ndarray) -> np.ndarray:
        """Propose a new state by adding Gaussian noise to current state."""
        noise = self.rng.normal(0.0, self.proposal_std, size=self._current_state.shape)
        q_proposed = self._current_state + noise
        q_proposed = np.clip(q_proposed, bounds[0], bounds[1])
        return q_proposed

    def _compute_log_acceptance(
        self,
        q_proposed: np.ndarray,
        q_current: np.ndarray,
        goal_belief: GoalBelief | None,
        obstacles: list[Polygon] | None,
        bounds: np.ndarray,
        sensor_config=None,
    ) -> float:
        """Compute log acceptance probability using Metropolis rule.

        Returns min(0, log π(q_proposed) - log π(q_current)).
        """
        log_pi_proposed = self._log_target_density(
            q_proposed, goal_belief, obstacles, bounds, sensor_config
        )
        log_pi_current = self._log_target_density(
            q_current, goal_belief, obstacles, bounds, sensor_config
        )

        return min(0.0, log_pi_proposed - log_pi_current)

    def _log_target_density(
        self,
        q: np.ndarray,
        goal_belief: GoalBelief | None,
        obstacles: list[Polygon] | None,
        bounds: np.ndarray,
        sensor_config=None,
    ) -> float:
        """Compute log of target distribution π(q) = exp(-J(q) / T).

        Returns log π(q) = -J(q) / T.
        """
        cost = self._compute_cost(q, goal_belief, obstacles, bounds, sensor_config)
        return -cost / self.temperature

    def _compute_cost(
        self,
        q: np.ndarray,
        goal_belief: GoalBelief | None,
        obstacles: list[Polygon] | None,
        bounds: np.ndarray,
        sensor_config=None,
    ) -> float:
        """Compute cost function with information gain term.

        J(q) = λ_goal * J_goal(q) + λ_clearance * J_clearance(q) - λ_info * InfoGain(q)

        Returns +∞ if q is out-of-bounds or colliding.
        """
        if not np.all(bounds[0] <= q) or not np.all(q <= bounds[1]):
            return _INFINITE_COST

        if obstacles is not None and not point_is_free(q, obstacles):
            return _INFINITE_COST

        if goal_belief is not None:
            goal_cost = self.lambda_goal * goal_belief.expected_distance(q)
        else:
            center = (bounds[0] + bounds[1]) / 2.0
            goal_cost = self.lambda_goal * np.linalg.norm(q - center)

        if obstacles is not None:
            clearance = self._compute_clearance(q, obstacles)
            clearance_cost = self.lambda_clearance / max(clearance, self.clearance_epsilon)
        else:
            clearance_cost = 0.0

        # Information gain term (active perception)
        info_gain_cost = 0.0
        if goal_belief is not None and sensor_config is not None:
            info_gain = goal_belief.predict_information_gain(q, sensor_config)
            info_gain_cost = -self.lambda_info * info_gain  # Negative because we want to reward high info gain

        total_cost = goal_cost + clearance_cost + info_gain_cost
        return max(total_cost, 0.0)  # Clamp to non-negative for numerical stability

    def _compute_clearance(self, q: np.ndarray, obstacles: list[Polygon]) -> float:
        """Compute minimum distance from point q to any obstacle."""
        point = Point(q[0], q[1])

        min_distance = float("inf")
        for obs in obstacles:
            dist = point.distance(obs.boundary)
            min_distance = min(min_distance, dist)

        return max(min_distance, 0.0)

    def _accept_proposal(self, log_acceptance: float) -> bool:
        """Decide whether to accept a proposal using Metropolis rule.

        Always accepts if log_acceptance >= 0, otherwise accepts with
        probability exp(log_acceptance).
        """
        if log_acceptance >= 0.0:
            return True

        acceptance_prob = np.exp(log_acceptance)
        return self.rng.random() < acceptance_prob

    def reset(self, seed: int) -> None:
        """Reset the sampler state, including the MCMC chain."""
        self.rng = np.random.default_rng(seed)
        self._current_state = None
        self._dimension = None
        self.sample_log = []
