"""Base sampler interface for RRT path planning.

This module defines the abstract base class that all samplers must implement,
ensuring a consistent interface for different sampling strategies.
"""

from abc import ABC, abstractmethod
from typing import List, Tuple

import numpy as np

# Type alias for sample log entries: (position, rejection_type, cost)
SampleLogEntry = Tuple[np.ndarray, str, float]


class BaseSampler(ABC):
    """
    Abstract base class for RRT samplers.

    All samplers must implement sample() and reset() methods to ensure
    compatibility with the RRT planner. This allows different sampling
    strategies (uniform, MCMC, etc.) to be swapped without changing
    the planner code.

    Subclasses should initialize self.sample_log = [] in their __init__.
    """

    def clear_sample_log(self) -> None:
        """Clear the sample log."""
        self.sample_log = []

    def get_sample_log(self) -> List[SampleLogEntry]:
        """Return the sample log, or empty list if logging not initialized."""
        return getattr(self, "sample_log", [])

    @abstractmethod
    def sample(self, bounds: np.ndarray, state: dict | None = None) -> np.ndarray:
        """
        Generate a random sample point in the configuration space.

        Args:
            bounds: Workspace limits with shape (2, D): [[min...], [max...]]
                    For 2D: [[x_min, y_min], [x_max, y_max]]
            state: Optional state dictionary containing context for adaptive sampling.
                   May include keys like 'goal_belief', 'env', 'current_tree_state', etc.
                   Samplers that don't use state (e.g., uniform) can ignore this parameter.

        Returns:
            Sample point as numpy array with shape (D,)
            For 2D: [x, y]
        """
        pass

    @abstractmethod
    def reset(self, seed: int) -> None:
        """
        Reset the random state of the sampler for reproducibility.

        Args:
            seed: Random seed value
        """
        pass
