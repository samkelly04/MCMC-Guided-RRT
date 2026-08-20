"""Comparative visualization functions for experimental results.

This module provides functions to generate publication-ready plots comparing
different agents across various metrics.
"""

from typing import List, Optional
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import patches as mpatches

from .loaders import (
    LoadedTrialResult,
    filter_by_agent,
    filter_by_complexity,
    filter_successful,
    filter_adaptive,
    get_unique_agents,
    get_unique_complexity_levels,
)


def plot_success_rates(
    results: List[LoadedTrialResult],
    save_path: str,
    figsize: tuple = (10, 6),
) -> None:
    """Plot success rates by agent and complexity level.

    Args:
        results: List of trial results.
        save_path: Path to save the figure.
        figsize: Figure size (width, height) in inches.
    """
    agents = get_unique_agents(results)
    complexity_levels = get_unique_complexity_levels(results)

    # Compute success rates
    success_rates = {}
    for agent in agents:
        success_rates[agent] = []
        for level in complexity_levels:
            filtered = filter_by_agent(results, agent)
            filtered = filter_by_complexity(filtered, level)
            if filtered:
                rate = sum(1 for r in filtered if r.success) / len(filtered)
                success_rates[agent].append(rate * 100)  # Convert to percentage
            else:
                success_rates[agent].append(0)

    # Create grouped bar chart
    fig, ax = plt.subplots(figsize=figsize)

    x = np.arange(len(complexity_levels))
    width = 0.35 if len(agents) == 2 else 0.25
    multiplier = 0

    for agent in agents:
        offset = width * multiplier
        ax.bar(x + offset, success_rates[agent], width, label=agent)
        multiplier += 1

    ax.set_ylabel('Success Rate (%)', fontsize=12)
    ax.set_xlabel('Complexity Level', fontsize=12)
    ax.set_title('Success Rate by Agent and Complexity', fontsize=14, fontweight='bold')
    ax.set_xticks(x + width * (len(agents) - 1) / 2)
    ax.set_xticklabels([l.capitalize() for l in complexity_levels])
    ax.legend(loc='best')
    ax.set_ylim(0, 105)
    ax.grid(axis='y', alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()


def plot_computation_time_comparison(
    results: List[LoadedTrialResult],
    save_path: str,
    figsize: tuple = (12, 6),
) -> None:
    """Plot computation time comparison across agents using box plots.

    Args:
        results: List of trial results.
        save_path: Path to save the figure.
        figsize: Figure size (width, height) in inches.
    """
    agents = get_unique_agents(results)
    complexity_levels = get_unique_complexity_levels(results)

    fig, axes = plt.subplots(1, len(complexity_levels), figsize=figsize, sharey=True)
    if len(complexity_levels) == 1:
        axes = [axes]

    for idx, level in enumerate(complexity_levels):
        ax = axes[idx]
        data_by_agent = []
        labels = []

        for agent in agents:
            filtered = filter_by_agent(results, agent)
            filtered = filter_by_complexity(filtered, level)
            times = [r.computation_time for r in filtered]
            data_by_agent.append(times)
            labels.append(agent)

        bp = ax.boxplot(data_by_agent, labels=labels, patch_artist=True)

        # Customize box colors
        colors = ['lightblue', 'lightcoral', 'lightgreen', 'lightyellow']
        for patch, color in zip(bp['boxes'], colors[:len(agents)]):
            patch.set_facecolor(color)

        ax.set_title(level.capitalize(), fontsize=12, fontweight='bold')
        ax.set_ylabel('Computation Time (s)' if idx == 0 else '', fontsize=11)
        ax.grid(axis='y', alpha=0.3)
        ax.tick_params(axis='x', rotation=15)

    fig.suptitle('Computation Time Comparison by Complexity Level', fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()


def plot_path_length_comparison(
    results: List[LoadedTrialResult],
    save_path: str,
    figsize: tuple = (12, 6),
) -> None:
    """Plot path length comparison for successful trials using box plots.

    Args:
        results: List of trial results.
        save_path: Path to save the figure.
        figsize: Figure size (width, height) in inches.
    """
    agents = get_unique_agents(results)
    complexity_levels = get_unique_complexity_levels(results)

    # Filter to successful trials only
    results = filter_successful(results)

    fig, axes = plt.subplots(1, len(complexity_levels), figsize=figsize, sharey=True)
    if len(complexity_levels) == 1:
        axes = [axes]

    for idx, level in enumerate(complexity_levels):
        ax = axes[idx]
        data_by_agent = []
        labels = []

        for agent in agents:
            filtered = filter_by_agent(results, agent)
            filtered = filter_by_complexity(filtered, level)
            lengths = [r.path_length for r in filtered]
            if not lengths:
                # Add empty list to maintain alignment, will show as missing data
                lengths = []
            data_by_agent.append(lengths)
            labels.append(agent)

        # Only plot if at least one agent has data
        if any(data_by_agent):
            bp = ax.boxplot(data_by_agent, labels=labels, patch_artist=True)

            # Customize box colors
            colors = ['lightblue', 'lightcoral', 'lightgreen', 'lightyellow']
            for patch, color in zip(bp['boxes'], colors[:len(agents)]):
                patch.set_facecolor(color)
        else:
            ax.text(0.5, 0.5, 'No successful trials',
                   ha='center', va='center', transform=ax.transAxes)

        ax.set_title(level.capitalize(), fontsize=12, fontweight='bold')
        ax.set_ylabel('Path Length' if idx == 0 else '', fontsize=11)
        ax.grid(axis='y', alpha=0.3)
        ax.tick_params(axis='x', rotation=15)

    fig.suptitle('Path Length Comparison (Successful Trials Only)', fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()


def plot_replans_by_complexity(
    results: List[LoadedTrialResult],
    save_path: str,
    figsize: tuple = (10, 6),
) -> None:
    """Plot average number of replans by complexity level (adaptive agents only).

    Args:
        results: List of trial results.
        save_path: Path to save the figure.
        figsize: Figure size (width, height) in inches.
    """
    # Filter to adaptive agents only
    results = filter_adaptive(results)

    if not results:
        print("Warning: No adaptive agent results found. Skipping replans plot.")
        return

    agents = get_unique_agents(results)
    complexity_levels = get_unique_complexity_levels(results)

    # Compute average replans
    avg_replans = {}
    std_replans = {}
    for agent in agents:
        avg_replans[agent] = []
        std_replans[agent] = []
        for level in complexity_levels:
            filtered = filter_by_agent(results, agent)
            filtered = filter_by_complexity(filtered, level)
            replans = [r.num_replans for r in filtered if r.num_replans is not None]
            if replans:
                avg_replans[agent].append(np.mean(replans))
                std_replans[agent].append(np.std(replans))
            else:
                avg_replans[agent].append(0)
                std_replans[agent].append(0)

    # Create bar chart with error bars
    fig, ax = plt.subplots(figsize=figsize)

    x = np.arange(len(complexity_levels))
    width = 0.35 if len(agents) == 2 else 0.25
    multiplier = 0

    for agent in agents:
        offset = width * multiplier
        ax.bar(
            x + offset,
            avg_replans[agent],
            width,
            yerr=std_replans[agent],
            label=agent,
            capsize=5,
        )
        multiplier += 1

    ax.set_ylabel('Average Number of Replans', fontsize=12)
    ax.set_xlabel('Complexity Level', fontsize=12)
    ax.set_title('Replanning Frequency by Complexity Level', fontsize=14, fontweight='bold')
    ax.set_xticks(x + width * (len(agents) - 1) / 2)
    ax.set_xticklabels([l.capitalize() for l in complexity_levels])
    ax.legend(loc='best')
    ax.grid(axis='y', alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()


def plot_belief_convergence(
    results: List[LoadedTrialResult],
    save_path: str,
    figsize: tuple = (10, 6),
) -> None:
    """Plot belief convergence (final belief accuracy) for adaptive agents.

    Args:
        results: List of trial results.
        save_path: Path to save the figure.
        figsize: Figure size (width, height) in inches.
    """
    # Filter to adaptive agents with belief convergence data
    results = [r for r in results if r.belief_convergence is not None]

    if not results:
        print("Warning: No belief convergence data found. Skipping plot.")
        return

    agents = get_unique_agents(results)
    complexity_levels = get_unique_complexity_levels(results)

    fig, axes = plt.subplots(1, len(complexity_levels), figsize=figsize, sharey=True)
    if len(complexity_levels) == 1:
        axes = [axes]

    for idx, level in enumerate(complexity_levels):
        ax = axes[idx]
        data_by_agent = []
        labels = []

        for agent in agents:
            filtered = filter_by_agent(results, agent)
            filtered = filter_by_complexity(filtered, level)
            convergence = [r.belief_convergence for r in filtered]
            data_by_agent.append(convergence)
            labels.append(agent)

        bp = ax.boxplot(data_by_agent, labels=labels, patch_artist=True)

        # Customize box colors
        colors = ['lightblue', 'lightcoral', 'lightgreen', 'lightyellow']
        for patch, color in zip(bp['boxes'], colors[:len(agents)]):
            patch.set_facecolor(color)

        ax.set_title(level.capitalize(), fontsize=12, fontweight='bold')
        ax.set_ylabel('Distance to True Goal' if idx == 0 else '', fontsize=11)
        ax.grid(axis='y', alpha=0.3)
        ax.tick_params(axis='x', rotation=15)

    fig.suptitle('Belief Convergence Accuracy (Distance from Final Belief to True Goal)',
                  fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()


def plot_all_comparative(
    results: List[LoadedTrialResult],
    output_dir: str,
) -> None:
    """Generate all comparative plots and save to output directory.

    Args:
        results: List of trial results.
        output_dir: Directory to save all plots.
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    print("Generating comparative plots...")

    plot_success_rates(results, str(output_path / "success_rates.png"))
    print("  ✓ Success rates")

    plot_computation_time_comparison(results, str(output_path / "computation_time.png"))
    print("  ✓ Computation time")

    plot_path_length_comparison(results, str(output_path / "path_length.png"))
    print("  ✓ Path length")

    plot_replans_by_complexity(results, str(output_path / "replans_by_complexity.png"))
    print("  ✓ Replanning frequency")

    plot_belief_convergence(results, str(output_path / "belief_convergence.png"))
    print("  ✓ Belief convergence")

    print(f"\nAll comparative plots saved to: {output_dir}")
