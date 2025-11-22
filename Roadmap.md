Collision Checker Function:
  Inputs: 
    1. A single point in 2D space
    2. List of Shapely Polygon objects representing walls/obstacles from the maze generator
  Returns: Boolean telling us whether or not the point is within any of the obstacles. 
    1. Returns TRUE if the point lies inside any of the obstacles 
    2. Returns FALSE if the point is not within any of the obstacles 
  Functionality: When running RRT we must know whether or not our randomly generated points lie in obstacles. 
  When running RRT the planning loop will call this function N times if N is the number of points we generate to build our tree.
  Uses Shapely geometry operations: Point.contains() for point collision, Polygon.intersects() for segment collision.

  Unit Testing: Comprehensive tests for point and segment collision detection with boundary cases.
  
Environment Generator:
  Generates maze-like environments using grid-based random walk algorithm.
  Returns: List of Shapely Polygon objects representing walls in continuous space.
  Inputs:
    - space_size: (width, height) of workspace
    - grid_size: (rows, cols) number of cells in grid
    - wall_thickness: Width of wall segments
    - seed: Random seed for reproducibility
  Features:
    - Guaranteed connectivity (at least one path exists)
    - Configurable dimensions
    - Continuous space (RRT samples from full space, not just grid cells) 


