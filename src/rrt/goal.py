"""Goal checking utilities for RRT path planning.

This module provides functions to check if the goal has been reached
and whether the goal is reachable from a given point. These utilities
are used by the RRT planner to determine when to terminate and how
to connect to the goal.
"""

import numpy as np
from .collision import segment_is_free


def is_goal_reached(point: np.ndarray, goal: np.ndarray, threshold: float) -> bool:
    """
    Check if a point is within the threshold distance of the goal.
    
    Args:
        point: Current point position [x, y]
        goal: Goal position [x, y]
        threshold: Maximum distance to consider goal reached
    
    Returns:
        True if point is within threshold of goal, False otherwise
    """
    # TODO: Implement goal reachability check
    # Calculate distance between point and goal
    # Return True if distance <= threshold
    distance = np.linalg.norm(point - goal)
    if distance <= threshold:
        return True
    else:
        return False 


def can_reach_goal_from(
point: np.ndarray, goal: np.ndarray, obstacles: list, threshold: float) -> bool:
    """
    Check if the goal can be reached from a point (collision-free segment).
    
    Args:
        point: Starting point position [x, y]
        goal: Goal position [x, y]
        obstacles: List of obstacle geometries (Shapely objects)
        threshold: Maximum distance to consider goal reached
    
    Returns:
        True if goal is within threshold AND the segment is collision-free
    """
    # TODO: Implement goal reachability check
    # First check if goal is within threshold using is_goal_reached()
    # Then check if segment from point to goal is collision-free
    # Return True only if both conditions are met
    raise NotImplementedError("can_reach_goal_from() not yet implemented")

