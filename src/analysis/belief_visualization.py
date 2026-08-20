"""Belief evolution visualization for adaptive agents.

This module provides functions to visualize how goal beliefs evolve during
adaptive navigation, including uncertainty ellipses and trajectory overlays.
"""

from typing import List, Dict, Any, Optional
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse
from matplotlib import patches as mpatches

from .loaders import LoadedTrialResult


def plot_belief_evolution_static(
    belief_history: List[Dict[str, Any]],
    true_goal: np.ndarray,
    executed_path: Optional[List[np.ndarray]] = None,
    obstacles: Optional[List] = None,
    save_path: Optional[str] = None,
    figsize: tuple = (10, 10),
    show_uncertainty: bool = True,
    uncertainty_sigma: float = 2.0,
) -> None:
    """Plot belief evolution with uncertainty ellipses (static plot).

    Args:
        belief_history: List of belief dictionaries with 'mean', 'covariance', etc.
        true_goal: True goal position (2D array).
        executed_path: Optional list of positions along executed trajectory.
        obstacles: Optional list of obstacle objects for visualization.
        save_path: Optional path to save figure. If None, displays plot.
        figsize: Figure size (width, height) in inches.
        show_uncertainty: Whether to show uncertainty ellipses.
        uncertainty_sigma: Number of standard deviations for ellipses (default: 2-sigma).
    """
    fig, ax = plt.subplots(figsize=figsize)

    # Plot executed path if provided
    if executed_path is not None and len(executed_path) > 0:
        path_array = np.array(executed_path)
        ax.plot(path_array[:, 0], path_array[:, 1], 'b-', linewidth=2,
                alpha=0.4, label='Executed Path')
        # Mark start position
        ax.plot(path_array[0, 0], path_array[0, 1], 'go', markersize=12,
                label='Start', zorder=5)

    # Plot belief trajectory (means over time)
    belief_means = np.array([b['mean'] for b in belief_history])
    ax.plot(belief_means[:, 0], belief_means[:, 1], 'r--', linewidth=2,
            alpha=0.7, label='Belief Trajectory', zorder=4)

    # Plot uncertainty ellipses at key points
    if show_uncertainty:
        # Sample beliefs to show (don't plot all if there are too many)
        num_beliefs = len(belief_history)
        if num_beliefs > 10:
            # Show initial, final, and evenly spaced intermediate beliefs
            indices = [0] + list(range(num_beliefs // 10, num_beliefs, num_beliefs // 10)) + [num_beliefs - 1]
            indices = sorted(set(indices))
        else:
            indices = range(num_beliefs)

        for idx in indices:
            belief = belief_history[idx]
            mean = belief['mean']
            cov = belief['covariance']

            # Compute eigenvalues and eigenvectors for ellipse
            eigenvalues, eigenvectors = np.linalg.eig(cov)
            angle = np.degrees(np.arctan2(eigenvectors[1, 0], eigenvectors[0, 0]))

            # Ellipse dimensions (uncertainty_sigma standard deviations)
            width = 2 * uncertainty_sigma * np.sqrt(eigenvalues[0])
            height = 2 * uncertainty_sigma * np.sqrt(eigenvalues[1])

            # Color by time (gradient from light to dark red)
            color_intensity = idx / max(1, num_beliefs - 1)
            color = (1.0, 1.0 - 0.7 * color_intensity, 1.0 - 0.7 * color_intensity)

            ellipse = Ellipse(
                xy=mean,
                width=width,
                height=height,
                angle=angle,
                facecolor=color,
                edgecolor='red',
                alpha=0.3,
                linewidth=1,
                zorder=2,
            )
            ax.add_patch(ellipse)

            # Mark belief mean
            ax.plot(mean[0], mean[1], 'r.', markersize=6, alpha=0.6, zorder=3)

    # Mark initial and final beliefs prominently
    if len(belief_history) > 0:
        initial_mean = belief_history[0]['mean']
        final_mean = belief_history[-1]['mean']

        ax.plot(initial_mean[0], initial_mean[1], 'rs', markersize=10,
                label='Initial Belief', zorder=5)
        ax.plot(final_mean[0], final_mean[1], 'r*', markersize=15,
                label='Final Belief', zorder=5)

    # Mark true goal
    ax.plot(true_goal[0], true_goal[1], 'g*', markersize=20,
            label='True Goal', zorder=6)

    # Plot obstacles if provided
    if obstacles is not None:
        for obs in obstacles:
            if hasattr(obs, 'polygon'):
                poly = obs.polygon
                x, y = poly.exterior.xy
                ax.fill(x, y, color='gray', alpha=0.5, zorder=1)
                ax.plot(x, y, 'k-', linewidth=0.5, zorder=1)

    ax.set_xlabel('X Position', fontsize=12)
    ax.set_ylabel('Y Position', fontsize=12)
    ax.set_title(f'Belief Evolution ({uncertainty_sigma}-sigma uncertainty ellipses)',
                 fontsize=14, fontweight='bold')
    ax.legend(loc='best', fontsize=10)
    ax.grid(True, alpha=0.3)
    ax.set_aspect('equal', adjustable='box')

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
    else:
        plt.show()


def plot_belief_evolution_from_trial(
    trial: LoadedTrialResult,
    true_goal: np.ndarray,
    save_path: Optional[str] = None,
    **kwargs,
) -> None:
    """Plot belief evolution from a trial result.

    Args:
        trial: LoadedTrialResult containing belief history.
        true_goal: True goal position.
        save_path: Optional path to save figure.
        **kwargs: Additional arguments passed to plot_belief_evolution_static.
    """
    if trial.belief_history is None or len(trial.belief_history) == 0:
        print(f"Warning: Trial has no belief history data")
        return

    plot_belief_evolution_static(
        belief_history=trial.belief_history,
        true_goal=true_goal,
        executed_path=None,  # Path not stored in trial results
        obstacles=None,  # Obstacles not stored in trial results
        save_path=save_path,
        **kwargs,
    )


def plot_entropy_evolution(
    entropy_history: List[float],
    num_replans: int,
    save_path: Optional[str] = None,
    figsize: tuple = (10, 6),
    entropy_threshold: Optional[float] = None,
) -> None:
    """Plot entropy evolution over time.

    Args:
        entropy_history: List of entropy values (trace of covariance) over time.
        num_replans: Number of replanning events.
        save_path: Optional path to save figure.
        figsize: Figure size (width, height) in inches.
        entropy_threshold: Optional entropy reduction threshold for replanning.
    """
    fig, ax = plt.subplots(figsize=figsize)

    steps = np.arange(len(entropy_history))
    ax.plot(steps, entropy_history, 'b-', linewidth=2, label='Entropy')

    # Mark replanning events (approximate locations)
    if num_replans > 0 and len(entropy_history) > 1:
        # Find approximate replan locations by detecting sharp increases in entropy
        entropy_array = np.array(entropy_history)
        entropy_diff = np.diff(entropy_array)

        # Replan likely occurred when entropy increased significantly
        replan_threshold = np.mean(entropy_diff) + 2 * np.std(entropy_diff)
        replan_indices = np.where(entropy_diff > replan_threshold)[0] + 1

        # Limit to num_replans
        replan_indices = replan_indices[:num_replans]

        for idx in replan_indices:
            ax.axvline(x=idx, color='red', linestyle='--', alpha=0.5, linewidth=1.5)

        if len(replan_indices) > 0:
            ax.axvline(x=replan_indices[0], color='red', linestyle='--',
                      alpha=0.5, linewidth=1.5, label='Replan Events')

    # Add threshold line if provided
    if entropy_threshold is not None:
        ax.axhline(y=entropy_threshold, color='orange', linestyle=':',
                  linewidth=2, label=f'Entropy Threshold ({entropy_threshold})')

    ax.set_xlabel('Step', fontsize=12)
    ax.set_ylabel('Entropy (trace of covariance)', fontsize=12)
    ax.set_title(f'Belief Entropy Evolution ({num_replans} replans)', fontsize=14, fontweight='bold')
    ax.legend(loc='best')
    ax.grid(True, alpha=0.3)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
    else:
        plt.show()


def plot_entropy_evolution_from_trial(
    trial: LoadedTrialResult,
    save_path: Optional[str] = None,
    **kwargs,
) -> None:
    """Plot entropy evolution from a trial result.

    Args:
        trial: LoadedTrialResult containing entropy history.
        save_path: Optional path to save figure.
        **kwargs: Additional arguments passed to plot_entropy_evolution.
    """
    if trial.entropy_history is None or len(trial.entropy_history) == 0:
        print(f"Warning: Trial has no entropy history data")
        return

    num_replans = trial.num_replans if trial.num_replans is not None else 0

    plot_entropy_evolution(
        entropy_history=trial.entropy_history,
        num_replans=num_replans,
        save_path=save_path,
        **kwargs,
    )


def plot_belief_convergence_over_time(
    belief_history: List[Dict[str, Any]],
    true_goal: np.ndarray,
    save_path: Optional[str] = None,
    figsize: tuple = (10, 6),
) -> None:
    """Plot distance from belief mean to true goal over time.

    Args:
        belief_history: List of belief dictionaries.
        true_goal: True goal position.
        save_path: Optional path to save figure.
        figsize: Figure size (width, height) in inches.
    """
    fig, ax = plt.subplots(figsize=figsize)

    # Compute distances over time
    distances = []
    for belief in belief_history:
        mean = belief['mean']
        distance = np.linalg.norm(mean - true_goal)
        distances.append(distance)

    steps = np.arange(len(distances))
    ax.plot(steps, distances, 'b-', linewidth=2)

    ax.set_xlabel('Step', fontsize=12)
    ax.set_ylabel('Distance from Belief to True Goal', fontsize=12)
    ax.set_title('Belief Convergence Over Time', fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
    else:
        plt.show()
