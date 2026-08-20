"""Abstract interface for maze environments."""

from abc import ABC, abstractmethod
from typing import List, Tuple, Optional
import numpy as np


class MazeEnvironment(ABC):
    """Abstract base class for maze environments.
    
    This interface defines the contract that all maze environments must implement
    to work with the evaluation framework. Environments can be grid-based or
    continuous, but must provide methods to query the maze structure and state.
    """
    
    @abstractmethod
    def get_start(self) -> np.ndarray:
        """Get the starting position in the maze.
        
        Returns:
            Starting position as a numpy array. For 2D mazes, shape is (2,).
            For grid-based mazes, this may be integer coordinates.
            For continuous mazes, this may be float coordinates.
        """
        pass
    
    @abstractmethod
    def get_goal(self) -> np.ndarray:
        """Get the goal position in the maze.
        
        Returns:
            Goal position as a numpy array. Shape matches get_start().
        """
        pass
    
    @abstractmethod
    def is_wall(self, position: np.ndarray) -> bool:
        """Check if a position contains a wall/obstacle.
        
        Args:
            position: Position to check, shape matches get_start().
        
        Returns:
            True if position contains a wall/obstacle, False if free space.
        """
        pass
    
    @abstractmethod
    def get_neighbors(self, position: np.ndarray) -> List[np.ndarray]:
        """Get valid neighboring positions from a given position.
        
        For grid-based mazes, this returns adjacent cells (4 or 8-connected).
        For continuous mazes, this may return reachable positions within a step size.
        
        Args:
            position: Current position to get neighbors from.
        
        Returns:
            List of neighboring positions that are valid (not walls and in bounds).
            Empty list if no valid neighbors exist.
        """
        pass
    
    @abstractmethod
    def get_bounds(self) -> np.ndarray:
        """Get the workspace bounds.
        
        Returns:
            Bounds array with shape (2, D) where D is dimensionality:
            [[min_x, min_y, ...], [max_x, max_y, ...]]
        """
        pass
    
    @abstractmethod
    def get_dimensions(self) -> Tuple[int, ...]:
        """Get the dimensions of the maze.
        
        For grid mazes, returns (rows, cols).
        For continuous mazes, may return workspace size or None.
        
        Returns:
            Tuple of dimension sizes.
        """
        pass
    
    @abstractmethod
    def is_valid_position(self, position: np.ndarray) -> bool:
        """Check if a position is valid (in bounds and not a wall).
        
        Args:
            position: Position to validate.
        
        Returns:
            True if position is valid (in bounds and free space), False otherwise.
        """
        pass
