"""Uniform random sampler for RRT path planning.

This module implements uniform random sampling in the configuration space,
which is the standard sampling strategy for basic RRT algorithms.
"""

import numpy as np
from .base import BaseSampler


class UniformSampler(BaseSampler):
    """
    Uniform random sampler that generates samples uniformly across
    the configuration space.
    
    This is the standard sampling strategy for basic RRT algorithms.
    Each sample is drawn independently from a uniform distribution
    over the workspace bounds.
    """

    def __init__(self, seed: int | None = None):
        """
        Initialize the uniform sampler.
        
        Args:
            seed: Optional random seed for reproducibility
        """
        self.rng = np.random.default_rng(seed)

    def sample(self, bounds: np.ndarray) -> np.ndarray:
        """
        Generate a uniformly random sample point.
        
        Args:
            bounds: Workspace limits with shape (2, D): [[min...], [max...]]
                    For 2D: [[x_min, y_min], [x_max, y_max]]
        
        Returns:
            Sample point as numpy array with shape (D,)
            For 2D: [x, y]
        """
        # Generate random point uniformly within bounds
        # bounds[0] = [min_x, min_y]
        # bounds[1] = [max_x, max_y]
        # uniform() generates a random value for each dimension between min and max
        return self.rng.uniform(bounds[0], bounds[1])

    def reset(self, seed: int) -> None:
        """
        Reset the random state with a new seed.
        
        Args:
            seed: Random seed value
        """
        self.rng = np.random.default_rng(seed)

