from shapely import box
from shapely.geometry import Polygon
import random
import numpy as np
from typing import List, Tuple


def generate_maze(
    space_size: Tuple[float, float] = (100, 100),
    grid_size: Tuple[int, int] = (20, 20),
    wall_thickness: float = 1.0,
    seed: int | None = None,
) -> List[Polygon]:
    """
    Generate maze-like environment using grid-based random walk algorithm.
    
    Creates a maze structure with guaranteed connectivity from start to goal
    by using a random walk algorithm to connect grid cells. Walls are placed
    between disconnected cells and converted to Shapely rectangles in continuous
    space. RRT sampling operates in continuous space, not limited to grid cells.
    
    Args:
        space_size: (width, height) of workspace in continuous coordinates.
                    Default: (100, 100)
        grid_size: (rows, cols) number of cells in the grid. Default: (20, 20)
        wall_thickness: Width of wall segments. Default: 1.0
        seed: Random seed for reproducibility. Default: None (non-deterministic)
    
    Returns:
        List of Shapely Polygon objects representing walls in continuous space.
        Each wall is a rectangular obstacle that can be used with collision
        detection functions.
    
    Example:
        >>> obstacles = generate_maze(space_size=(100, 100), grid_size=(20, 20), seed=42)
        >>> len(obstacles)  # Number of wall segments
        150
    """
    # Set random seed for reproducibility if provided
    if seed is not None:
        random.seed(seed)
    
    # Calculate cell dimensions from workspace and grid size
    cell_width = space_size[0] / grid_size[0]
    cell_height = space_size[1] / grid_size[1]
    
    rows, cols = grid_size
    
    # Initialize: track which adjacent cells are connected (walls removed)
    # Use set of tuples to represent connected pairs: {(cell1, cell2), ...}
    # where cell1 < cell2 lexicographically for consistency
    connected_pairs = set()
    visited = set()
    
    # Random walk algorithm to create maze structure
    # Start from a random cell
    start_row = random.randint(0, rows - 1)
    start_col = random.randint(0, cols - 1)
    current = (start_row, start_col)
    visited.add(current)
    stack = [current]  # Stack for backtracking
    
    # Continue random walk until we've visited a significant portion of cells
    # or until we can't find unvisited neighbors
    target_visited = rows * cols  # Try to visit all cells for full connectivity
    
    while len(visited) < target_visited:
        # Get neighbors of current cell
        neighbors = []
        row, col = current
        
        # Check all four directions: North, South, East, West
        if row > 0:
            neighbors.append((row - 1, col))  # North
        if row < rows - 1:
            neighbors.append((row + 1, col))  # South
        if col > 0:
            neighbors.append((row, col - 1))  # West
        if col < cols - 1:
            neighbors.append((row, col + 1))  # East
        
        # Filter to unvisited neighbors
        unvisited_neighbors = [n for n in neighbors if n not in visited]
        
        if unvisited_neighbors:
            # Choose random unvisited neighbor
            next_cell = random.choice(unvisited_neighbors)
            
            # Mark cells as connected (remove wall between them)
            cell_pair = tuple(sorted([current, next_cell]))
            connected_pairs.add(cell_pair)
            
            # Move to next cell
            visited.add(next_cell)
            stack.append(next_cell)
            current = next_cell
        else:
            # Backtrack if no unvisited neighbors
            if stack:
                stack.pop()
                if stack:
                    current = stack[-1]
                else:
                    break
            else:
                break
    
    # Convert disconnected cell pairs to wall segments
    obstacles = []
    half_thickness = wall_thickness / 2.0
    
    # Check all vertical walls (between cells in same row, different columns)
    for row in range(rows):
        for col in range(cols - 1):
            cell1 = (row, col)
            cell2 = (row, col + 1)
            cell_pair = tuple(sorted([cell1, cell2]))
            
            if cell_pair not in connected_pairs:
                # Create vertical wall between these cells
                wall_x = (col + 1) * cell_width
                wall_y_min = row * cell_height
                wall_y_max = (row + 1) * cell_height
                
                wall = box(
                    wall_x - half_thickness,
                    wall_y_min,
                    wall_x + half_thickness,
                    wall_y_max
                )
                obstacles.append(wall)
    
    # Check all horizontal walls (between cells in same column, different rows)
    for row in range(rows - 1):
        for col in range(cols):
            cell1 = (row, col)
            cell2 = (row + 1, col)
            cell_pair = tuple(sorted([cell1, cell2]))
            
            if cell_pair not in connected_pairs:
                # Create horizontal wall between these cells
                wall_y = (row + 1) * cell_height
                wall_x_min = col * cell_width
                wall_x_max = (col + 1) * cell_width
                
                wall = box(
                    wall_x_min,
                    wall_y - half_thickness,
                    wall_x_max,
                    wall_y + half_thickness
                )
                obstacles.append(wall)
    
    # Add outer boundary walls (optional - comment out if you want open boundaries)
    # For now, we'll add them to create a complete enclosed maze
    boundary_thickness = wall_thickness
    
    # Bottom wall
    obstacles.append(box(
        -boundary_thickness,
        -boundary_thickness,
        space_size[0] + boundary_thickness,
        0
    ))
    
    # Top wall
    obstacles.append(box(
        -boundary_thickness,
        space_size[1],
        space_size[0] + boundary_thickness,
        space_size[1] + boundary_thickness
    ))
    
    # Left wall
    obstacles.append(box(
        -boundary_thickness,
        -boundary_thickness,
        0,
        space_size[1] + boundary_thickness
    ))
    
    # Right wall
    obstacles.append(box(
        space_size[0],
        -boundary_thickness,
        space_size[0] + boundary_thickness,
        space_size[1] + boundary_thickness
    ))
    
    return obstacles   





