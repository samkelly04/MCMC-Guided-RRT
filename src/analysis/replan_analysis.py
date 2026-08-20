"""Replanning behavior analysis for adaptive agents.

This module provides functions to analyze when and why replanning occurs,
including entropy-based triggers and correlations with success.
"""

from typing import List, Dict, Optional
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

from .loaders import (
    LoadedTrialResult,
    filter_adaptive,
    filter_by_complexity,
    filter_successful,
    get_unique_complexity_levels,
)


def analyze_replan_triggers(results: List[LoadedTrialResult]) -> Dict:
    """Analyze replanning triggers and statistics.

    Args:
        results: List of trial results.

    Returns:
        Dictionary containing replanning statistics.
    """
    # Filter to adaptive agents only
    adaptive_results = filter_adaptive(results)

    if not adaptive_results:
        return {}

    # Compute statistics
    total_replans = sum(r.num_replans for r in adaptive_results if r.num_replans is not None)
    num_trials = len(adaptive_results)
    avg_replans = total_replans / num_trials if num_trials > 0 else 0

    # Replans by success
    successful = filter_successful(adaptive_results)
    failed = [r for r in adaptive_results if not r.success]

    avg_replans_success = (
        sum(r.num_replans for r in successful if r.num_replans is not None) / len(successful)
        if successful else 0
    )
    avg_replans_failure = (
        sum(r.num_replans for r in failed if r.num_replans is not None) / len(failed)
        if failed else 0
    )

    # Replans by complexity
    complexity_levels = get_unique_complexity_levels(adaptive_results)
    avg_replans_by_complexity = {}
    for level in complexity_levels:
        level_results = filter_by_complexity(adaptive_results, level)
        avg_replans_by_complexity[level] = (
            sum(r.num_replans for r in level_results if r.num_replans is not None) / len(level_results)
            if level_results else 0
        )

    # Compute entropy reduction per replan (if possible)
    entropy_reductions = []
    for result in adaptive_results:
        if result.entropy_history and result.num_replans and result.num_replans > 0:
            initial_entropy = result.entropy_history[0]
            final_entropy = result.entropy_history[-1]
            total_reduction = initial_entropy - final_entropy
            avg_reduction_per_replan = total_reduction / result.num_replans
            entropy_reductions.append(avg_reduction_per_replan)

    return {
        'total_trials': num_trials,
        'total_replans': total_replans,
        'avg_replans_per_trial': avg_replans,
        'avg_replans_success': avg_replans_success,
        'avg_replans_failure': avg_replans_failure,
        'avg_replans_by_complexity': avg_replans_by_complexity,
        'avg_entropy_reduction_per_replan': np.mean(entropy_reductions) if entropy_reductions else 0,
        'std_entropy_reduction_per_replan': np.std(entropy_reductions) if entropy_reductions else 0,
    }


def plot_replan_frequency_histogram(
    results: List[LoadedTrialResult],
    save_path: Optional[str] = None,
    figsize: tuple = (10, 6),
) -> None:
    """Plot histogram of replan counts across trials.

    Args:
        results: List of trial results.
        save_path: Optional path to save figure.
        figsize: Figure size (width, height) in inches.
    """
    # Filter to adaptive agents only
    adaptive_results = filter_adaptive(results)

    if not adaptive_results:
        print("Warning: No adaptive agent results found. Skipping replan histogram.")
        return

    replan_counts = [r.num_replans for r in adaptive_results if r.num_replans is not None]

    fig, ax = plt.subplots(figsize=figsize)

    ax.hist(replan_counts, bins=range(0, max(replan_counts) + 2),
            edgecolor='black', alpha=0.7, color='steelblue')

    ax.set_xlabel('Number of Replans', fontsize=12)
    ax.set_ylabel('Frequency', fontsize=12)
    ax.set_title('Distribution of Replanning Events', fontsize=14, fontweight='bold')
    ax.grid(axis='y', alpha=0.3)

    # Add statistics text
    mean_replans = np.mean(replan_counts)
    median_replans = np.median(replan_counts)
    ax.axvline(mean_replans, color='red', linestyle='--', linewidth=2,
               label=f'Mean: {mean_replans:.1f}')
    ax.axvline(median_replans, color='orange', linestyle='--', linewidth=2,
               label=f'Median: {median_replans:.1f}')
    ax.legend()

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
    else:
        plt.show()


def plot_replans_vs_success(
    results: List[LoadedTrialResult],
    save_path: Optional[str] = None,
    figsize: tuple = (8, 6),
) -> None:
    """Plot relationship between number of replans and success rate.

    Args:
        results: List of trial results.
        save_path: Optional path to save figure.
        figsize: Figure size (width, height) in inches.
    """
    # Filter to adaptive agents only
    adaptive_results = filter_adaptive(results)

    if not adaptive_results:
        print("Warning: No adaptive agent results found. Skipping replans vs success plot.")
        return

    successful = filter_successful(adaptive_results)
    failed = [r for r in adaptive_results if not r.success]

    success_replans = [r.num_replans for r in successful if r.num_replans is not None]
    failure_replans = [r.num_replans for r in failed if r.num_replans is not None]

    fig, ax = plt.subplots(figsize=figsize)

    data = [success_replans, failure_replans]
    labels = ['Success', 'Failure']
    colors = ['green', 'red']

    bp = ax.boxplot(data, labels=labels, patch_artist=True)

    for patch, color in zip(bp['boxes'], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.6)

    ax.set_ylabel('Number of Replans', fontsize=12)
    ax.set_title('Replanning Frequency by Trial Outcome', fontsize=14, fontweight='bold')
    ax.grid(axis='y', alpha=0.3)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
    else:
        plt.show()


def plot_entropy_reduction_analysis(
    results: List[LoadedTrialResult],
    save_path: Optional[str] = None,
    figsize: tuple = (10, 6),
) -> None:
    """Plot entropy reduction over trials.

    Args:
        results: List of trial results.
        save_path: Optional path to save figure.
        figsize: Figure size (width, height) in inches.
    """
    # Filter to adaptive agents with entropy history
    adaptive_results = [r for r in results if r.entropy_history and len(r.entropy_history) > 0]

    if not adaptive_results:
        print("Warning: No entropy history data found. Skipping entropy reduction plot.")
        return

    initial_entropies = []
    final_entropies = []
    reductions = []

    for result in adaptive_results:
        initial = result.entropy_history[0]
        final = result.entropy_history[-1]
        reduction = initial - final

        initial_entropies.append(initial)
        final_entropies.append(final)
        reductions.append(reduction)

    fig, axes = plt.subplots(1, 2, figsize=figsize)

    # Plot initial vs final entropy
    ax1 = axes[0]
    ax1.scatter(initial_entropies, final_entropies, alpha=0.6, color='steelblue')
    ax1.plot([0, max(initial_entropies)], [0, max(initial_entropies)],
             'r--', alpha=0.5, label='No reduction line')
    ax1.set_xlabel('Initial Entropy', fontsize=11)
    ax1.set_ylabel('Final Entropy', fontsize=11)
    ax1.set_title('Initial vs Final Entropy', fontsize=12, fontweight='bold')
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # Plot entropy reduction distribution
    ax2 = axes[1]
    ax2.hist(reductions, bins=20, edgecolor='black', alpha=0.7, color='steelblue')
    ax2.axvline(np.mean(reductions), color='red', linestyle='--', linewidth=2,
                label=f'Mean: {np.mean(reductions):.2f}')
    ax2.set_xlabel('Entropy Reduction', fontsize=11)
    ax2.set_ylabel('Frequency', fontsize=11)
    ax2.set_title('Entropy Reduction Distribution', fontsize=12, fontweight='bold')
    ax2.legend()
    ax2.grid(axis='y', alpha=0.3)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
    else:
        plt.show()


def generate_replan_analysis_report(
    results: List[LoadedTrialResult],
    save_path: str,
) -> None:
    """Generate text report analyzing replanning behavior.

    Args:
        results: List of trial results.
        save_path: Path to save the report.
    """
    stats = analyze_replan_triggers(results)

    if not stats:
        print("Warning: No adaptive agent data found. Skipping replan analysis report.")
        return

    report_lines = []
    report_lines.append("="*80)
    report_lines.append("REPLANNING BEHAVIOR ANALYSIS")
    report_lines.append("="*80)
    report_lines.append("")

    report_lines.append(f"Total Trials (Adaptive Agents): {stats['total_trials']}")
    report_lines.append(f"Total Replanning Events: {stats['total_replans']}")
    report_lines.append(f"Average Replans per Trial: {stats['avg_replans_per_trial']:.2f}")
    report_lines.append("")

    report_lines.append("-"*80)
    report_lines.append("REPLANNING BY TRIAL OUTCOME")
    report_lines.append("-"*80)
    report_lines.append(f"Average Replans (Successful Trials): {stats['avg_replans_success']:.2f}")
    report_lines.append(f"Average Replans (Failed Trials): {stats['avg_replans_failure']:.2f}")
    report_lines.append("")

    report_lines.append("-"*80)
    report_lines.append("REPLANNING BY COMPLEXITY LEVEL")
    report_lines.append("-"*80)
    for level, avg in stats['avg_replans_by_complexity'].items():
        report_lines.append(f"{level.upper()}: {avg:.2f} replans/trial")
    report_lines.append("")

    report_lines.append("-"*80)
    report_lines.append("ENTROPY REDUCTION")
    report_lines.append("-"*80)
    report_lines.append(f"Average Entropy Reduction per Replan: {stats['avg_entropy_reduction_per_replan']:.4f}")
    report_lines.append(f"Std Dev: {stats['std_entropy_reduction_per_replan']:.4f}")
    report_lines.append("")

    report_lines.append("="*80)

    with open(save_path, 'w') as f:
        f.write('\n'.join(report_lines))

    print(f"Replan analysis report saved to: {save_path}")


def plot_all_replan_analysis(
    results: List[LoadedTrialResult],
    output_dir: str,
) -> None:
    """Generate all replanning analysis plots.

    Args:
        results: List of trial results.
        output_dir: Directory to save plots.
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    print("Generating replanning analysis plots...")

    plot_replan_frequency_histogram(results, str(output_path / "replan_frequency_histogram.png"))
    print("  ✓ Replan frequency histogram")

    plot_replans_vs_success(results, str(output_path / "replans_vs_success.png"))
    print("  ✓ Replans vs success")

    plot_entropy_reduction_analysis(results, str(output_path / "entropy_reduction_analysis.png"))
    print("  ✓ Entropy reduction analysis")

    generate_replan_analysis_report(results, str(output_path / "replan_analysis_report.txt"))
    print("  ✓ Replan analysis report")

    print(f"\nReplanning analysis saved to: {output_dir}")
