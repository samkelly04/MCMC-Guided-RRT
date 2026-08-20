"""Concrete implementation of MazeEnvironment interface for continuous mazes."""

from typing import List, Tuple, Optional
import numpy as np
from shapely.geometry import Polygon

from ..core.environment import MazeEnvironment
from ..config.experiment_config import MazeGenerationConfig
from ..rrt.collision import point_is_free
from .generator import generate_maze


class ContinuousMazeEnvironment(MazeEnvironment):
    """Concrete implementation of MazeEnvironment for continuous-space mazes.
    
    This class wraps the generate_maze() function and provides a complete
    implementation of the MazeEnvironment interface. It handles:
    - Maze generation using the existing generator
    - Start/goal position management
    - Collision checking using existing collision detection
    - Neighbor queries for continuous space
    
    Example:
        >>> from src.config import MazeGenerationConfig
        >>> config = MazeGenerationConfig(
        ...     space_size=(100, 100),
        ...     grid_size=(10, 10),
        ...     seed=42
        ... )
        >>> env = ContinuousMazeEnvironment(config, start=(10, 10), goal=(90, 90))
        >>> env.get_start()
        array([10., 10.])
        >>> env.is_wall(np.array([50, 50]))
        False
    """
    
    def __init__(
        self,
        maze_config: MazeGenerationConfig,
        start: Optional[np.ndarray] = None,
        goal: Optional[np.ndarray] = None,
        neighbor_step_size: float = 3.0,
    ):
        """Initialize a continuous maze environment.
        
        Args:
            maze_config: Configuration for maze generation.
            start: Starting position. If None, will be generated automatically.
            goal: Goal position. If None, will be generated automatically.
            neighbor_step_size: Step size for get_neighbors() in continuous space.
                               Default: 3.0 (matches typical RRT step size).
        """
        self.maze_config = maze_config
        self.neighbor_step_size = neighbor_step_size
        
        # Generate obstacles using existing generator
        self.obstacles: List[Polygon] = generate_maze(
            space_size=maze_config.space_size,
            grid_size=maze_config.grid_size,
            wall_thickness=maze_config.wall_thickness,
            seed=maze_config.seed,
        )
        
        # Set bounds
        self.bounds = np.array([
            [0.0, 0.0],
            list(maze_config.space_size)
        ], dtype=float)
        
        # Generate or use provided start/goal positions
        if start is None:
            self.start = self._generate_valid_position()
        else:
            self.start = np.asarray(start, dtype=float).flatten()
            if not self.is_valid_position(self.start):
                raise ValueError(f"Start position {self.start} is not valid (collision or out of bounds)")
        
        if goal is None:
            self.goal = self._generate_valid_position()
        else:
            self.goal = np.asarray(goal, dtype=float).flatten()
            if not self.is_valid_position(self.goal):
                raise ValueError(f"Goal position {self.goal} is not valid (collision or out of bounds)")
    
    def get_start(self) -> np.ndarray:
        """Get the starting position in the maze.
        
        Returns:
            Starting position as numpy array with shape (2,).
        """
        return self.start.copy()
    
    def get_goal(self) -> np.ndarray:
        """Get the goal position in the maze.
        
        Returns:
            Goal position as numpy array with shape (2,).
        """
        return self.goal.copy()
    
    def is_wall(self, position: np.ndarray) -> bool:
        """Check if a position contains a wall/obstacle.
        
        Args:
            position: Position to check, shape (2,).
        
        Returns:
            True if position contains a wall/obstacle, False if free space.
        """
        return not point_is_free(position, self.obstacles)
    
    def get_neighbors(self, position: np.ndarray) -> List[np.ndarray]:
        """Get valid neighboring positions from a given position.
        
        For continuous space, this returns positions reachable within
        neighbor_step_size in 8 directions (N, NE, E, SE, S, SW, W, NW).
        
        Args:
            position: Current position to get neighbors from.
        
        Returns:
            List of neighboring positions that are valid (not walls and in bounds).
            Empty list if no valid neighbors exist.
        """
        position = np.asarray(position, dtype=float).flatten()
        step = self.neighbor_step_size
        
        # 8-directional neighbors (N, NE, E, SE, S, SW, W, NW)
        directions = [
            (0, step),      # North
            (step, step),   # Northeast
            (step, 0),      # East
            (step, -step),  # Southeast
            (0, -step),     # South
            (-step, -step), # Southwest
            (-step, 0),     # West
            (-step, step),  # Northwest
        ]
        
        neighbors = []
        for dx, dy in directions:
            neighbor = position + np.array([dx, dy])
            
            # Check if neighbor is valid
            if self.is_valid_position(neighbor):
                neighbors.append(neighbor)
        
        return neighbors
    
    def get_bounds(self) -> np.ndarray:
        """Get the workspace bounds.
        
        Returns:
            Bounds array with shape (2, 2):
            [[min_x, min_y], [max_x, max_y]]
        """
        return self.bounds.copy()
    
    def get_dimensions(self) -> Tuple[int, ...]:
        """Get the dimensions of the maze.
        
        Returns:
            Tuple (rows, cols) from grid_size.
        """
        return tuple(self.maze_config.grid_size)
    
    def is_valid_position(self, position: np.ndarray) -> bool:
        """Check if a position is valid (in bounds and not a wall).
        
        Args:
            position: Position to validate, shape (2,).
        
        Returns:
            True if position is valid (in bounds and free space), False otherwise.
        """
        position = np.asarray(position, dtype=float).flatten()
        
        # Check bounds
        if not np.all(self.bounds[0] <= position) or not np.all(position <= self.bounds[1]):
            return False
        
        # Check collision
        return point_is_free(position, self.obstacles)
    
    def get_obstacles(self) -> List[Polygon]:
        """Get the list of obstacle polygons.
        
        This is a convenience method for agents that need direct access
        to obstacles (e.g., RRT planner).
        
        Returns:
            List of Shapely Polygon objects representing walls/obstacles.
        """
        return self.obstacles
    
    def get_seed(self) -> Optional[int]:
        """Get the random seed used to generate this maze.
        
        Returns:
            Random seed, or None if no seed was used.
        """
        return self.maze_config.seed
    
    def _generate_valid_position(self) -> np.ndarray:
        """Generate a random valid (collision-free) position in the maze.
        
        This is used when start/goal are not provided. It samples random
        positions until finding one that is collision-free.
        
        Returns:
            Valid position as numpy array with shape (2,).
        """
        import random
        
        max_attempts = 1000
        for _ in range(max_attempts):
            # Sample random position within bounds
            x = random.uniform(self.bounds[0][0], self.bounds[1][0])
            y = random.uniform(self.bounds[0][1], self.bounds[1][1])
            position = np.array([x, y])
            
            if self.is_valid_position(position):
                return position
        
        # Fallback: try grid cell centers if random sampling fails
        cell_width = self.maze_config.space_size[0] / self.maze_config.grid_size[0]
        cell_height = self.maze_config.space_size[1] / self.maze_config.grid_size[1]
        
        for row in range(self.maze_config.grid_size[0]):
            for col in range(self.maze_config.grid_size[1]):
                x = (col + 0.5) * cell_width
                y = (row + 0.5) * cell_height
                position = np.array([x, y])
                
                if self.is_valid_position(position):
                    return position
        
        # Last resort: return center of workspace
        return np.array([
            self.maze_config.space_size[0] / 2.0,
            self.maze_config.space_size[1] / 2.0
        ])
