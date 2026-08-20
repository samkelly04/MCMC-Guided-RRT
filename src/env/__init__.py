"""Environment generation module for RRT path planning.

This module handles the creation of maze-like environments with obstacles
for path planning algorithms. It uses Shapely for geometric operations
and grid-based random walk algorithms to ensure connectivity.
"""

from .generator import generate_maze
from .maze_environment import ContinuousMazeEnvironment

__all__ = ["generate_maze", "ContinuousMazeEnvironment"]
