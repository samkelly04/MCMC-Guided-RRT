"""Nearest neighbor search for RRT path planning.

This module provides efficient nearest neighbor search functionality
for the RRT algorithm, finding the closest tree node to a given sample point.
"""

import numpy as np
from .tree import Tree


def find_nearest(tree: Tree, sample_point: np.ndarray) -> int:
    """
    Find the index of the nearest node in the tree to a given sample point.
    
    Args:
        tree: Tree object containing nodes to search
        sample_point: 2D point [x, y] to find nearest neighbor for
        
    Returns:
        Index of the nearest node in the tree
        
    Raises:
        ValueError: If tree is empty
    """
    # Get all tree positions
    tree_positions = tree.get_all_positions()
    
    # Handle empty tree case
    if len(tree_positions) == 0:
        raise ValueError("Tree is empty")
    
    # Calculate squared distances for each node
    distances = []
    for i, node_pos in enumerate(tree_positions):
        squared_dist = np.linalg.norm(node_pos - sample_point) ** 2
        distances.append(squared_dist)
    
    # Find the index of the minimum distance
    nearest_idx = np.argmin(distances)
    return nearest_idx
