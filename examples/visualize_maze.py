#!/usr/bin/env python
"""Standalone script to visualize maze generation.

This script can be run directly to generate and visualize mazes.
Usage: python examples/visualize_maze.py
"""

import sys
from pathlib import Path

# Add project root to path so we can import src
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import matplotlib.pyplot as plt
from matplotlib.patches import Polygon as MplPolygon
from src.env.generator import generate_maze


def main():
    """Generate and visualize a maze."""
    print("Generating maze...")
    
    # Generate maze
    obstacles = generate_maze(
        space_size=(100, 100),
        grid_size=(20, 20),
        wall_thickness=1.0,
        seed=42
    )
    
    print(f"✓ Generated maze with {len(obstacles)} obstacles")
    
    # Verify obstacles are Shapely polygons
    for obs in obstacles:
        assert hasattr(obs, 'exterior'), "Each obstacle should be a Shapely geometry"
        assert hasattr(obs, 'bounds'), "Each obstacle should have bounds"
    
    # Create visualization
    fig, ax = plt.subplots(1, 1, figsize=(10, 10))
    
    # Plot each obstacle (wall) as a filled rectangle
    for obs in obstacles:
        # Get exterior coordinates
        x, y = obs.exterior.xy
        # Convert to matplotlib polygon
        poly = MplPolygon(list(zip(x, y)), facecolor='black', edgecolor='black')
        ax.add_patch(poly)
    
    # Set axis limits and labels
    ax.set_xlim(-5, 105)
    ax.set_ylim(-5, 105)
    ax.set_aspect('equal')
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_title(f'Maze Environment (seed=42)\n{len(obstacles)} wall segments')
    ax.grid(True, alpha=0.3)
    
    # Save figure
    output_path = project_root / 'maze_visualization_standalone.png'
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"✓ Visualization saved to {output_path}")
    print(f"✓ Maze generation successful!")


if __name__ == '__main__':
    main()

