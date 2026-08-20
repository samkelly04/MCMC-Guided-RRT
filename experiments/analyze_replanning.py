#!/usr/bin/env python3
"""Script to analyze replanning behavior in adaptive agents.

This script generates comprehensive analysis of when and why replanning
occurs, including correlations with success and entropy reduction.

Usage:
    python experiments/analyze_replanning.py results/batch_comparison/trials_*.json
    python experiments/analyze_replanning.py results/batch_comparison/trials_*.json --output-dir results/replan_analysis
"""

import argparse
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.analysis.loaders import load_trials, filter_adaptive
from src.analysis.replan_analysis import (
    plot_all_replan_analysis,
    analyze_replan_triggers,
)


def main():
    parser = argparse.ArgumentParser(
        description='Analyze replanning behavior in adaptive agents.'
    )
    parser.add_argument(
        'results_file',
        type=str,
        help='Path to JSON file containing trial results'
    )
    parser.add_argument(
        '--output-dir',
        type=str,
        default='results/replan_analysis',
        help='Directory to save analysis outputs (default: results/replan_analysis)'
    )

    args = parser.parse_args()

    # Validate input file
    if not Path(args.results_file).exists():
        print(f"Error: Results file not found: {args.results_file}")
        sys.exit(1)

    print("="*80)
    print("REPLANNING BEHAVIOR ANALYSIS")
    print("="*80)
    print(f"\nLoading results from: {args.results_file}")

    # Load results
    try:
        results = load_trials(args.results_file)
        print(f"✓ Loaded {len(results)} trial results")
    except Exception as e:
        print(f"Error loading results: {e}")
        sys.exit(1)

    # Filter to adaptive agents
    adaptive_results = filter_adaptive(results)
    print(f"✓ Found {len(adaptive_results)} adaptive agent trials")

    if not adaptive_results:
        print("Error: No adaptive agent trials found")
        sys.exit(1)

    # Create output directory
    output_path = Path(args.output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    print(f"✓ Output directory: {output_path}")

    # Print summary statistics
    print("\n" + "-"*80)
    print("REPLANNING STATISTICS")
    print("-"*80)
    stats = analyze_replan_triggers(adaptive_results)

    if stats:
        print(f"Total Trials: {stats['total_trials']}")
        print(f"Total Replans: {stats['total_replans']}")
        print(f"Average Replans per Trial: {stats['avg_replans_per_trial']:.2f}")
        print(f"\nBy Outcome:")
        print(f"  Successful Trials: {stats['avg_replans_success']:.2f} replans/trial")
        print(f"  Failed Trials: {stats['avg_replans_failure']:.2f} replans/trial")
        print(f"\nBy Complexity:")
        for level, avg in stats['avg_replans_by_complexity'].items():
            print(f"  {level.upper()}: {avg:.2f} replans/trial")
        print(f"\nEntropy Reduction:")
        print(f"  Avg per Replan: {stats['avg_entropy_reduction_per_replan']:.4f}")
        print(f"  Std Dev: {stats['std_entropy_reduction_per_replan']:.4f}")

    # Generate all plots
    print("\n" + "-"*80)
    try:
        plot_all_replan_analysis(adaptive_results, str(output_path))
    except Exception as e:
        print(f"Error generating plots: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

    print("\n" + "="*80)
    print("ANALYSIS COMPLETE!")
    print("="*80)
    print(f"\nResults saved to: {output_path}")
    print("\nGenerated files:")
    print("  - replan_frequency_histogram.png")
    print("  - replans_vs_success.png")
    print("  - entropy_reduction_analysis.png")
    print("  - replan_analysis_report.txt")
    print()


if __name__ == '__main__':
    main()
