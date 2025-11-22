"""Visualization test for maze generator.

This test generates mazes and visualizes them to verify the generator
creates viable environments. Run with: pytest tests/test_generator_visualization.py -v
"""

import matplotlib.pyplot as plt
from matplotlib.patches import Polygon as MplPolygon
from src.env.generator import generate_maze


def test_generate_maze_visualization():
    """
    Generate and visualize a maze to verify it creates viable environments.
    
    This test will:
    1. Generate a maze with specified parameters
    2. Create a matplotlib visualization showing walls
    3. Verify obstacles are Shapely polygons
    4. Display the plot (requires display or will save to file)
    """
    # Generate maze
    obstacles = generate_maze(
        space_size=(100, 100),
        grid_size=(20, 20),
        wall_thickness=1.0,
        seed=42
    )
    
    # Verify we got obstacles
    assert len(obstacles) > 0, "Should generate at least some obstacles"
    
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
    
    # Save figure instead of showing (for CI/CD and headless environments)
    plt.savefig('maze_visualization.png', dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"✓ Generated maze with {len(obstacles)} obstacles")
    print(f"✓ Visualization saved to maze_visualization.png")


def test_generate_maze_different_sizes():
    """
    Test maze generation with different grid sizes and visualize them.
    """
    test_cases = [
        {"space_size": (50, 50), "grid_size": (10, 10), "seed": 1, "name": "Small"},
        {"space_size": (100, 100), "grid_size": (20, 20), "seed": 2, "name": "Medium"},
        {"space_size": (100, 100), "grid_size": (30, 30), "seed": 3, "name": "Dense"},
    ]
    
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    
    for idx, test_case in enumerate(test_cases):
        obstacles = generate_maze(
            space_size=test_case["space_size"],
            grid_size=test_case["grid_size"],
            wall_thickness=1.0,
            seed=test_case["seed"]
        )
        
        ax = axes[idx]
        
        # Plot obstacles
        for obs in obstacles:
            x, y = obs.exterior.xy
            poly = MplPolygon(list(zip(x, y)), facecolor='black', edgecolor='black')
            ax.add_patch(poly)
        
        # Configure subplot
        width, height = test_case["space_size"]
        ax.set_xlim(-5, width + 5)
        ax.set_ylim(-5, height + 5)
        ax.set_aspect('equal')
        ax.set_title(f'{test_case["name"]}\n{test_case["grid_size"]} grid, {len(obstacles)} walls')
        ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('maze_comparison.png', dpi=150, bbox_inches='tight')
    plt.close()
    
    print("✓ Generated and visualized mazes with different sizes")
    print("✓ Comparison saved to maze_comparison.png")

