"""RRT Planner implementation.

This module provides the main RRT planning algorithm that combines
sampling, nearest neighbor search, steering, and collision checking
to find a path from start to goal.
"""

import numpy as np
from typing import List, Optional
from shapely.geometry import Polygon

from .tree import Tree
from .nearest import find_nearest
from .steer import steer
from .collision import segment_is_free, point_is_free
from .goal import is_goal_reached
from ..samplers.base import BaseSampler


class RRTPlanner:
    """
    RRT (Rapidly-exploring Random Tree) path planner.
    
    This planner grows a tree incrementally from a start position toward
    a goal by sampling random points, finding the nearest tree node,
    steering toward the sample, and adding collision-free nodes.
    
    The algorithm terminates when the goal is reached or maximum iterations
    are exceeded.
    """
    
    def __init__(
        self,
        start: np.ndarray,
        goal: np.ndarray,
        obstacles: List[Polygon],
        sampler: BaseSampler,
        step_size: float,
        goal_threshold: float,
        bounds: np.ndarray,
        max_iterations: int = 10000,
    ):
        """
        Initialize the RRT planner.
        
        Args:
            start: Starting position [x, y]
            goal: Goal position [x, y]
            obstacles: List of Shapely Polygon obstacles
            sampler: Sampler instance (e.g., UniformSampler)
            step_size: Maximum step size for tree expansion
            goal_threshold: Maximum distance to consider goal reached
            bounds: Workspace limits with shape (2, D): [[min...], [max...]]
                    For 2D: [[x_min, y_min], [x_max, y_max]]
            max_iterations: Maximum number of iterations before giving up
        """
        self.start = np.array(start, dtype=float)
        self.goal = np.array(goal, dtype=float)
        self.obstacles = obstacles
        self.sampler = sampler
        self.step_size = step_size
        self.goal_threshold = goal_threshold
        self.bounds = np.array(bounds, dtype=float)
        self.max_iterations = max_iterations
        
        # Validate inputs
        self._validate_inputs()
    
    def _validate_inputs(self) -> None:
        """Validate that inputs are reasonable."""
        # Check that start and goal are in bounds
        if not point_is_free(self.start, self.obstacles):
            raise ValueError("Start position is in collision with obstacles")
        
        # Check bounds shape
        if self.bounds.shape != (2, len(self.start)):
            raise ValueError(
                f"Bounds must have shape (2, {len(self.start)}), "
                f"got {self.bounds.shape}"
            )
        
        # Check that start and goal are within bounds
        if not np.all(self.bounds[0] <= self.start) or not np.all(self.start <= self.bounds[1]):
            raise ValueError("Start position is outside bounds")
        
        if not np.all(self.bounds[0] <= self.goal) or not np.all(self.goal <= self.bounds[1]):
            raise ValueError("Goal position is outside bounds")
    
    def plan(self) -> Optional[List[np.ndarray]]:
        """
        Execute the RRT planning algorithm.
        
        Returns:
            List of positions forming the path from start to goal, or None if
            no path found within max_iterations.
        """
        # Initialize tree with start position
        tree = Tree(self.start)
        
        # Main planning loop
        for iteration in range(self.max_iterations):
            # Step 1: Sample a random point in the workspace
            q_rand = self.sampler.sample(self.bounds)
            
            # Step 2: Find the nearest node in the tree to this sample
            nearest_idx = find_nearest(tree, q_rand)
            q_near = tree.get_node_position(nearest_idx)
            
            # Step 3: Steer from nearest node toward the sample
            # This limits movement to step_size and clamps to bounds
            q_new = steer(q_near, q_rand, self.step_size, self.bounds)
            
            # Step 4: Check if the edge from q_near to q_new is collision-free
            if segment_is_free(q_near, q_new, self.obstacles):
                # Safe! Add the new node to the tree
                new_idx = tree.add_node(nearest_idx, q_new)
                
                # Step 5: Check if we can reach the goal from this new node
                # First check if goal is close enough
                if is_goal_reached(q_new, self.goal, self.goal_threshold):
                    # Check if the segment to goal is collision-free
                    if segment_is_free(q_new, self.goal, self.obstacles):
                        # Success! Add goal node and extract path
                        goal_idx = tree.add_node(new_idx, self.goal)
                        path = self._extract_path(tree, goal_idx)
                        return path
            # If collision or goal not reached, continue to next iteration
        
        # Failed to find path within max_iterations
        return None
    
    def _extract_path(self, tree: Tree, goal_node_idx: int) -> List[np.ndarray]:
        """
        Extract the path from start to goal by backtracking through the tree.
        
        Args:
            tree: The RRT tree containing the path
            goal_node_idx: Index of the goal node in the tree
        
        Returns:
            List of positions from start to goal (including both endpoints)
        """
        # Get path from goal back to root (following parent pointers)
        path_to_root = tree.get_path_to_root(goal_node_idx)
        
        # Reverse to get path from start to goal
        path = list(reversed(path_to_root))
        
        return path

