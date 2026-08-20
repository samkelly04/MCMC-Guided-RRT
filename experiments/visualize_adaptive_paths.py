#!/usr/bin/env python3
"""Visualize AdaptiveMCMC navigation paths - successful and failed cases.

NOTE ON CONFIG DRIFT: this script does not use the same settings as the headline
experiment in run_batch_comparison.py. It runs with max_iterations=10000,
max_steps=300, adaptive_iterations disabled, and start/goal pinned to opposite
corners at a 12-unit margin. Under that configuration several easy mazes fail at
step 0 with nothing executable, which does NOT match the 100% easy success rate
reported in results/analysis_v4/. Align these parameters with
experiments/run_batch_comparison.py before reading anything into its output.
"""

import sys
from pathlib import Path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon as MplPolygon
from matplotlib.collections import PatchCollection

from src.config import MazeGenerationConfig
from src.env.maze_environment import ContinuousMazeEnvironment
from src.samplers.adaptive_goal_mcmc import AdaptiveGoalMCMCSampler
from src.rrt.goal_belief import GoalBelief
from src.rrt.goal_sensor import GoalObservationModel, GoalSensorConfig
from src.rrt.navigation import run_closed_loop_navigation


def plot_maze(ax, env, title=""):
    """Plot the maze with obstacles."""
    # Plot obstacles
    patches = []
    for obs in env.get_obstacles():
        coords = np.array(obs.exterior.coords)
        patch = MplPolygon(coords, closed=True)
        patches.append(patch)
    
    collection = PatchCollection(patches, facecolor='#2d3436', edgecolor='#636e72', linewidth=0.5)
    ax.add_collection(collection)
    
    # Set bounds
    bounds = env.get_bounds()
    ax.set_xlim(bounds[0][0] - 2, bounds[1][0] + 2)
    ax.set_ylim(bounds[0][1] - 2, bounds[1][1] + 2)
    ax.set_aspect('equal')
    ax.set_title(title, fontsize=12, fontweight='bold')
    ax.set_xlabel('X')
    ax.set_ylabel('Y')


def plot_navigation_result(ax, env, result, show_belief_evolution=True):
    """Plot the navigation result including path and belief evolution."""
    plot_maze(ax, env)
    
    start = env.get_start()
    goal = env.get_goal()
    
    # Plot start and goal
    ax.plot(start[0], start[1], 'go', markersize=12, label='Start', zorder=10)
    ax.plot(goal[0], goal[1], 'r*', markersize=15, label='True Goal', zorder=10)
    
    # Plot executed path
    if result['executed_path'] and len(result['executed_path']) > 1:
        path = np.array(result['executed_path'])
        ax.plot(path[:, 0], path[:, 1], 'b-', linewidth=2, alpha=0.8, label='Executed Path')
        ax.plot(path[-1, 0], path[-1, 1], 'b^', markersize=10, label='Final Position')
    
    # Plot belief evolution
    if show_belief_evolution and result['belief_history']:
        beliefs = result['belief_history']
        colors = plt.cm.Oranges(np.linspace(0.3, 1.0, len(beliefs)))
        
        for i, belief in enumerate(beliefs):
            mean = belief.mean
            # Plot belief mean
            ax.plot(mean[0], mean[1], 'o', color=colors[i], markersize=6, alpha=0.7)
            
            # Draw uncertainty ellipse (simplified - just a circle based on trace)
            std = np.sqrt(np.trace(belief.covariance) / 2)
            circle = plt.Circle((mean[0], mean[1]), std, fill=False, 
                               color=colors[i], linestyle='--', alpha=0.5)
            ax.add_patch(circle)
        
        # Connect belief means with arrows
        if len(beliefs) > 1:
            for i in range(len(beliefs) - 1):
                m1 = beliefs[i].mean
                m2 = beliefs[i + 1].mean
                ax.annotate('', xy=m2, xytext=m1,
                           arrowprops=dict(arrowstyle='->', color='orange', alpha=0.5))
    
    # Add result info
    status = "✓ SUCCESS" if result['success'] else "✗ FAILED"
    color = 'green' if result['success'] else 'red'
    info = f"{status}\nSteps: {result['num_steps']}, Replans: {result['num_replans']}"
    info += f"\nFinal dist: {result['distance_to_goal']:.1f}"
    
    ax.text(0.02, 0.98, info, transform=ax.transAxes, fontsize=9,
            verticalalignment='top', fontfamily='monospace',
            bbox=dict(boxstyle='round', facecolor='white', alpha=0.8),
            color=color)


def run_and_visualize(seed, grid_size, title_suffix=""):
    """Run navigation on a maze and return the result for visualization."""
    config = MazeGenerationConfig(
        space_size=(100, 100), 
        grid_size=grid_size, 
        seed=seed
    )
    
    # Try to create environment with corner positions
    margin = 12.0
    try:
        env = ContinuousMazeEnvironment(
            config,
            start=np.array([margin, margin]),
            goal=np.array([100 - margin, 100 - margin])
        )
    except ValueError:
        env = ContinuousMazeEnvironment(config)
    
    # Create components
    sampler = AdaptiveGoalMCMCSampler(
        proposal_std=8.0,
        temperature=10.0,
        lambda_goal=1.0,
        lambda_clearance=0.0,
        lambda_info=2.0,
        uniform_mixing_rate=0.30,
        seed=42,
    )
    
    sensor = GoalObservationModel(
        GoalSensorConfig(far_std=10.0, near_std=0.5, distance_scale=50.0, seed=42)
    )
    
    initial_belief = GoalBelief(
        mean=np.array([50.0, 50.0]),
        covariance=np.eye(2) * 400.0,
    )
    
    # Run navigation
    result = run_closed_loop_navigation(
        sampler=sampler,
        sensor=sensor,
        true_goal=env.get_goal(),
        start_pose=env.get_start(),
        initial_belief=initial_belief,
        obstacles=env.get_obstacles(),
        bounds=env.get_bounds(),
        step_size=5.0,
        goal_threshold=5.0,
        goal_tolerance=8.0,
        max_steps=300,
        execution_horizon=10,
        entropy_threshold=5.0,
        max_iterations=10000,
        min_steps_between_replans=3,
    )
    
    title = f"Seed {seed} ({grid_size[0]}×{grid_size[1]} grid) {title_suffix}"
    return env, result, title


def main():
    """Generate visualization of successful and failed navigation cases."""
    
    # Test cases: (seed, grid_size, expected_outcome)
    test_cases = [
        # Easy mazes (6x6) - should succeed
        (100, (6, 6), "Easy"),
        (101, (6, 6), "Easy"),
        (102, (6, 6), "Easy"),
        
        # Medium mazes (8x8) - mixed results
        (103, (8, 8), "Medium"),
        (104, (8, 8), "Medium"),
        (105, (8, 8), "Medium"),
        
        # Hard mazes (10x10) - challenging
        (106, (10, 10), "Hard"),
        (107, (10, 10), "Hard"),
        (108, (10, 10), "Hard"),
    ]
    
    # Create figure
    fig, axes = plt.subplots(3, 3, figsize=(15, 15))
    fig.suptitle('AdaptiveMCMC with Receding-Horizon Navigation\n'
                 '(Orange circles: belief evolution, Blue line: executed path)',
                 fontsize=14, fontweight='bold')
    
    for idx, (seed, grid_size, difficulty) in enumerate(test_cases):
        row = idx // 3
        col = idx % 3
        ax = axes[row, col]
        
        print(f"Running seed {seed} ({difficulty})...", end=" ", flush=True)
        env, result, title = run_and_visualize(seed, grid_size, f"[{difficulty}]")
        print("✓" if result['success'] else "✗")
        
        plot_navigation_result(ax, env, result)
        ax.set_title(title, fontsize=10)
    
    # Add legend to first subplot
    axes[0, 0].legend(loc='lower right', fontsize=8)
    
    plt.tight_layout()
    plt.savefig('results/adaptive_mcmc_paths.png', dpi=150, bbox_inches='tight')
    print(f"\nSaved visualization to: results/adaptive_mcmc_paths.png")
    plt.show()


if __name__ == "__main__":
    main()

