"""Generate publication-quality figure showing AdaptiveMCMCAgent failure mode.

This script visualizes why the agent fails to make progress in dense mazes:
- Shows maze structure with start/goal
- Illustrates the RRT tree growth limitation
- Demonstrates partial path that makes no progress
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.patches import Ellipse, FancyArrowPatch
from shapely.geometry import Polygon

from src.env.generator import generate_maze
from src.samplers.adaptive_goal_mcmc import AdaptiveGoalMCMCSampler
from src.rrt.planner import RRTPlanner
from src.rrt.goal_belief import GoalBelief
from src.rrt.goal_sensor import GoalObservationModel, GoalSensorConfig


def visualize_failure_case(seed=160, grid_size=(8, 8), output_path="failure_mode_analysis.png"):
    """Create visualization showing why RRT fails in dense mazes.

    Args:
        seed: Maze seed to use (default: 160)
        grid_size: Maze grid size (default: (8, 8) for medium complexity)
        output_path: Path to save figure
    """
    # Generate maze
    complexity = "medium" if grid_size == (8, 8) else "hard" if grid_size == (10, 10) else "custom"
    print(f"Generating {complexity} maze {grid_size} (seed={seed})...")
    obstacles = generate_maze(
        space_size=(100, 100),
        grid_size=grid_size,
        wall_thickness=1.0,
        seed=seed,
    )

    # Configuration matching experiment
    bounds = np.array([[0.0, 0.0], [100.0, 100.0]])
    start = np.array([15.0, 15.0])
    goal = np.array([85.0, 85.0])

    # Create initial belief (high uncertainty)
    initial_belief = GoalBelief(
        mean=np.array([50.0, 50.0]),
        covariance=np.eye(2) * 400.0,
    )

    # Take 5 observations to localize goal (as in navigation.py)
    sensor_config = GoalSensorConfig(
        far_std=10.0,
        near_std=0.5,
        distance_scale=20.0,
    )
    sensor = GoalObservationModel(sensor_config)

    belief = initial_belief.copy()
    for _ in range(5):
        obs, obs_cov = sensor.observe(start, goal)
        belief.update(obs, obs_cov)

    print(f"After 5 observations:")
    print(f"  Belief mean: {belief.mean}")
    print(f"  True goal: {goal}")
    print(f"  Belief error: {np.linalg.norm(belief.mean - goal):.2f}")
    print(f"  Distance to goal: {np.linalg.norm(start - goal):.2f}")

    # Run RRT planning (will fail to make progress)
    print(f"\nRunning RRT with 15,000 iterations...")
    sampler = AdaptiveGoalMCMCSampler(
        proposal_std=8.0,
        temperature=10.0,
        lambda_goal=1.0,
        lambda_clearance=0.0,
        lambda_info=2.0,
        uniform_mixing_rate=0.30,
        seed=seed,
    )

    planner = RRTPlanner(
        start=start,
        goal=belief.mean,  # Plan toward belief mean
        obstacles=obstacles,
        sampler=sampler,
        step_size=1.0,
        goal_threshold=0.5,
        bounds=bounds,
        max_iterations=15000,
        goal_belief=belief,
        sensor_config=sensor_config,
    )

    partial_path = planner.plan(return_partial=True)

    if partial_path is not None and len(partial_path) > 0:
        path_end = partial_path[-1]
        distance_to_target = np.linalg.norm(path_end - belief.mean)
        print(f"  Partial path length: {len(partial_path)} waypoints")
        print(f"  Path end: {path_end}")
        print(f"  Distance to target: {distance_to_target:.2f}")
        print(f"  Progress made: {np.linalg.norm(start - goal) - distance_to_target:.2f} units")
    else:
        print(f"  No path returned!")

    # Create publication-quality figure
    print(f"\nCreating visualization...")
    fig, ax = plt.subplots(1, 1, figsize=(10, 10))

    # Draw obstacles
    for obs in obstacles:
        if isinstance(obs, Polygon):
            x, y = obs.exterior.xy
            ax.fill(x, y, color='gray', alpha=0.7, edgecolor='black', linewidth=0.5)

    # Draw grid overlay to show cell structure
    for i in range(11):
        ax.axhline(i * 10, color='lightgray', linewidth=0.3, linestyle='--', alpha=0.5)
        ax.axvline(i * 10, color='lightgray', linewidth=0.3, linestyle='--', alpha=0.5)

    # Draw belief uncertainty (2-sigma ellipse)
    eigenvalues, eigenvectors = np.linalg.eig(belief.covariance)
    angle = np.degrees(np.arctan2(eigenvectors[1, 0], eigenvectors[0, 0]))
    width, height = 2 * 2 * np.sqrt(eigenvalues)  # 2-sigma

    ellipse = Ellipse(
        xy=belief.mean,
        width=width,
        height=height,
        angle=angle,
        facecolor='blue',
        alpha=0.2,
        edgecolor='blue',
        linewidth=2,
        label='Goal belief (2σ)',
    )
    ax.add_patch(ellipse)

    # Draw partial path (if any)
    if partial_path is not None and len(partial_path) > 1:
        path_array = np.array(partial_path)
        ax.plot(
            path_array[:, 0],
            path_array[:, 1],
            'o-',
            color='orange',
            linewidth=2,
            markersize=4,
            label=f'RRT partial path ({len(partial_path)} nodes)',
        )
        # Mark path end
        ax.plot(
            path_array[-1, 0],
            path_array[-1, 1],
            'X',
            color='orange',
            markersize=15,
            markeredgecolor='black',
            markeredgewidth=1.5,
            label='Best node reached',
        )

    # Draw start and goal
    ax.plot(start[0], start[1], 'go', markersize=15, label='Start [15, 15]',
            markeredgecolor='black', markeredgewidth=2)
    ax.plot(goal[0], goal[1], 'r*', markersize=20, label='True goal [85, 85]',
            markeredgecolor='black', markeredgewidth=1.5)
    ax.plot(belief.mean[0], belief.mean[1], 'b^', markersize=12,
            label=f'Belief mean [{belief.mean[0]:.1f}, {belief.mean[1]:.1f}]',
            markeredgecolor='black', markeredgewidth=1.5)

    # Add distance annotation
    ax.annotate(
        '',
        xy=goal,
        xytext=start,
        arrowprops=dict(
            arrowstyle='<->',
            color='red',
            lw=2,
            linestyle='--',
            alpha=0.6,
        ),
    )

    # Add text annotation for distance
    mid_point = (start + goal) / 2 + np.array([5, 5])
    ax.text(
        mid_point[0],
        mid_point[1],
        f'd = {np.linalg.norm(start - goal):.1f} units\n(~7 grid cells)',
        fontsize=11,
        bbox=dict(boxstyle='round', facecolor='white', alpha=0.8, edgecolor='red'),
        ha='center',
    )

    # Add problem description annotation
    if partial_path is not None and len(partial_path) > 0:
        progress = np.linalg.norm(path_end - start)
        problem_text = (
            f"Problem: RRT with 15,000 iterations\n"
            f"• Returned {len(partial_path)} waypoints\n"
            f"• Progress: {progress:.1f} units ({progress/np.linalg.norm(start-goal)*100:.1f}%)\n"
            f"• Still {distance_to_target:.1f} units from target\n"
            f"• After removing start: {max(0, len(partial_path)-1)} waypoints left\n"
            f"→ Navigation STALLS (nothing to execute)"
        )
    else:
        problem_text = (
            f"Problem: RRT with 15,000 iterations\n"
            f"• Returned NO path (complete failure)\n"
            f"• Cannot make ANY progress\n"
            f"→ Navigation immediately STALLS"
        )

    ax.text(
        50, 5,
        problem_text,
        fontsize=10,
        bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.9, edgecolor='red', linewidth=2),
        ha='center',
        va='bottom',
        family='monospace',
    )

    # Formatting
    ax.set_xlim(-5, 105)
    ax.set_ylim(-5, 105)
    ax.set_aspect('equal')
    ax.set_xlabel('X (units)', fontsize=12, fontweight='bold')
    ax.set_ylabel('Y (units)', fontsize=12, fontweight='bold')
    ax.set_title(
        'AdaptiveMCMCAgent Failure Mode: Dense Maze with Long Distance\n'
        f'{complexity.capitalize()} Complexity ({grid_size[0]}×{grid_size[1]} grid, seed={seed})',
        fontsize=14,
        fontweight='bold',
    )
    ax.legend(loc='upper right', fontsize=10, framealpha=0.9)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"\n✓ Figure saved to: {output_path}")
    plt.close()


if __name__ == '__main__':
    # Use hard complexity (10×10 grid) with seed 160
    # Shows interesting behavior: some progress but insufficient for success
    print("Generating publication figure for AdaptiveMCMCAgent failure mode...")
    visualize_failure_case(seed=160, grid_size=(10, 10), output_path="failure_mode_analysis.png")
