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
from .goal_belief import GoalBelief
from .goal_sensor import GoalSensorConfig
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
        goal_belief: Optional[GoalBelief] = None,
        sensor_config: Optional[GoalSensorConfig] = None,
    ):
        """
        Initialize the RRT planner.

        Args:
            start: Starting position [x, y]
            goal: Goal position [x, y]. If goal_belief is provided, this is
                   overridden by goal_belief.mean for adaptive planning.
            obstacles: List of Shapely Polygon obstacles
            sampler: Sampler instance (e.g., UniformSampler, AdaptiveGoalMCMCSampler)
            step_size: Maximum step size for tree expansion
            goal_threshold: Maximum distance to consider goal reached
            bounds: Workspace limits with shape (2, D): [[min...], [max...]]
                    For 2D: [[x_min, y_min], [x_max, y_max]]
            max_iterations: Maximum number of iterations before giving up
            goal_belief: Optional GoalBelief for adaptive goal-directed planning.
                        When provided, the planner targets goal_belief.mean instead
                        of the fixed goal parameter.
            sensor_config: Optional GoalSensorConfig for information-theoretic
                          sampling and active perception.
        """
        self.start = np.array(start, dtype=float)
        self.obstacles = obstacles
        self.sampler = sampler
        self.step_size = step_size
        self.goal_threshold = goal_threshold
        self.bounds = np.array(bounds, dtype=float)
        self.max_iterations = max_iterations
        self.goal_belief = goal_belief
        self.sensor_config = sensor_config

        # Always use the explicitly provided goal parameter
        # The goal_belief is used by samplers for guidance, not as the planning target
        self.goal = np.array(goal, dtype=float)
        self.segment_log: list = []

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

        # Validate goal (self.goal is already set to goal_belief.mean if belief provided)
        if not np.all(self.bounds[0] <= self.goal) or not np.all(self.goal <= self.bounds[1]):
            raise ValueError("Goal position is outside bounds")
    
    def plan(self, return_partial: bool = True) -> Optional[List[np.ndarray]]:
        """
        Execute the RRT planning algorithm.

        Args:
            return_partial: If True, return partial path to best node when full
                          path cannot be found. If False, return None on failure.

        Returns:
            List of positions forming the path from start to goal, or if
            return_partial=True and no complete path found, returns partial path
            to the node closest to the goal (best effort). Returns None only if
            return_partial=False and no path found.
        """
        # Initialize tree with start position
        tree = Tree(self.start)

        # Track the best node (closest to goal) for fallback
        best_node_idx = 0  # Start node
        best_distance_to_goal = np.linalg.norm(self.start - self.goal)

        # Main planning loop
        for iteration in range(self.max_iterations):
            # Step 1: Sample a random point in the workspace
            # Construct state dict for adaptive samplers
            state = None
            if self.goal_belief is not None or self.sensor_config is not None:
                state = {}
                if self.goal_belief is not None:
                    state['goal_belief'] = self.goal_belief
                if self.obstacles is not None:
                    state['obstacles'] = self.obstacles
                if self.sensor_config is not None:
                    state['sensor_config'] = self.sensor_config

            q_rand = self.sampler.sample(self.bounds, state=state)

            # Step 2: Find the nearest node in the tree to this sample
            nearest_idx = find_nearest(tree, q_rand)
            q_near = tree.get_node_position(nearest_idx)

            # Step 3: Steer from nearest node toward the sample
            # This limits movement to step_size and clamps to bounds
            q_new = steer(q_near, q_rand, self.step_size, self.bounds)

            # Step 4: Check if the edge from q_near to q_new is collision-free
            seg_free = segment_is_free(q_near, q_new, self.obstacles)
            self.segment_log.append((q_near.copy(), q_new.copy(), seg_free))

            if seg_free:
                # Safe! Add the new node to the tree
                new_idx = tree.add_node(nearest_idx, q_new)

                # Update best node tracker
                distance_to_goal = np.linalg.norm(q_new - self.goal)
                if distance_to_goal < best_distance_to_goal:
                    best_node_idx = new_idx
                    best_distance_to_goal = distance_to_goal

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

        # Failed to find complete path within max_iterations
        if return_partial:
            # Return partial path to best node (closest to goal)
            partial_path = self._extract_path(tree, best_node_idx)
            return partial_path
        else:
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

