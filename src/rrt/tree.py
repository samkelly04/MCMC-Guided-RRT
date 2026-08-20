import numpy as np
from typing import List, Optional


class Tree:
    """
    RRT Tree data structure for storing nodes and parent-child relationships.
    
    This class manages the core tree structure used in RRT path planning.
    It stores node positions and maintains parent-child relationships for
    efficient path reconstruction.
    """
    
    def __init__(self, root_position: np.ndarray):
        """
        Initialize tree with a root node.
        
        Args:
            root_position: 2D position of the root node [x, y]
        """
        self.positions = np.array([root_position])
        self.parents = np.array([-1])
    
    def add_node(self, parent_idx: int, position: np.ndarray) -> int:
        """
        Add a new node to the tree.
        
        Args:
            parent_idx: Index of the parent node (-1 for root)
            position: 2D position of the new node [x, y]
            
        Returns:
            Index of the newly added node
            
        Raises:
            ValueError: If parent_idx is invalid
        """
        self.positions = np.vstack((self.positions, position))
        # Parents should be 1D array, use append or concatenate
        self.parents = np.append(self.parents, parent_idx)
        return len(self.positions) - 1
    
    def get_node_position(self, idx: int) -> np.ndarray:
        """
        Get the position of a node by its index.
        
        Args:
            idx: Node index
            
        Returns:
            2D position [x, y]
            
        Raises:
            IndexError: If idx is out of bounds
        """
        return self.positions[idx]
    
    def get_parent(self, idx: int) -> int:
        """
        Get the parent index of a node.
        
        Args:
            idx: Node index
            
        Returns:
            Parent node index (-1 for root)
            
        Raises:
            IndexError: If idx is out of bounds
        """
        if idx < 0 or idx >= len(self.parents):
            raise IndexError("Index out of bounds")
        # Ensure parents is 1D for indexing
        parents_1d = self.parents.flatten() if self.parents.ndim > 1 else self.parents
        return int(parents_1d[idx])
    
    def get_path_to_root(self, node_idx: int) -> List[np.ndarray]:
        """
        Get the path from a node back to the root.
        
        Args:
            node_idx: Starting node index
            
        Returns:
            List of positions from node to root (including both endpoints)
            
        Raises:
            IndexError: If node_idx is out of bounds
        """
        path = []
        curr_idx = node_idx
        # Ensure parents is 1D for indexing
        parents_1d = self.parents.flatten() if self.parents.ndim > 1 else self.parents
        while curr_idx != -1:
            path.append(self.positions[curr_idx])
            curr_idx = int(parents_1d[curr_idx])
        
        return path

    def size(self) -> int:
        """
        Get the number of nodes in the tree.
        
        Returns:
            Number of nodes
        """
        return len(self.positions)
    
    def get_all_positions(self) -> np.ndarray:
        """
        Get all node positions for visualization or analysis.
        
        Returns:
            Array of shape (n_nodes, 2) containing all positions
        """
        return self.positions
    
    def _validate_index(self, idx: int) -> None:
        """
        Validate that an index is within bounds.
        
        Args:
            idx: Index to validate
            
        Raises:
            IndexError: If idx is out of bounds
        """
        if idx < 0 or idx >= len(self.positions):
            raise IndexError("Index out of bounds")
