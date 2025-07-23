Collision Checker Function:
  Inputs: 
    1. A single point in 2D space
    2. Dictionary of rectangular obstacles, where we have information about each given by xmin, xmax, ymin, ymax 
  Returns: Boolean telling us whether or not the point is within any of the rectangular obstacles. 
    1. Returns TRUE if the point lies inside any of the rectangular obstacles 
    2. Returns FALSE is the point is not within any of the rectangular obstacles 
  Functionality: When running RRT we must know whether or not our randomly generated points lie in the. 
  When running RRT the planning loop will call this function N times if N is the number of points we generate to build our tree.
  Must understand the geometry details. --> Should know whether we are using Shapely or just Tuples.

  Unit Testing: 
  
Environment Generator:
  Inputs: 


