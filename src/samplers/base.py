"""Base sampler interface for RRT path planning.

This module defines the abstract base class that all samplers must implement,
ensuring a consistent interface for different sampling strategies.
"""

from abc import ABC, abstractmethod
import numpy as np


class BaseSampler(ABC):
    """
    Abstract base class for RRT samplers.
    
    All samplers must implement sample() and reset() methods to ensure
    compatibility with the RRT planner. This allows different sampling
    strategies (uniform, MCMC, etc.) to be swapped without changing
    the planner code.
    """

    @abstractmethod
    def sample(self, bounds: np.ndarray) -> np.ndarray:
        """
        Generate a random sample point in the configuration space.
        
        Args:
            bounds: Workspace limits with shape (2, D): [[min...], [max...]]
                    For 2D: [[x_min, y_min], [x_max, y_max]]
        
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

