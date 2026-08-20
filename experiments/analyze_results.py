#!/usr/bin/env python3
"""Main analysis script for experimental results.

This script loads trial results from JSON files, generates comparative plots,
and performs statistical significance testing.

Usage:
    python experiments/analyze_results.py results/batch_comparison/trials_*.json
    python experiments/analyze_results.py results/batch_comparison/trials_*.json --output-dir results/analysis
"""

import argparse
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.analysis.loaders import load_trials
from src.analysis.comparative_plots import plot_all_comparative
from src.analysis.statistical_tests import generate_statistical_report


def main():
    parser = argparse.ArgumentParser(
        description='Analyze experimental results and generate plots and statistics.'
    )
    parser.add_argument(
        'results_file',
        type=str,
        help='Path to JSON file containing trial results'
    )
    parser.add_argument(
        '--output-dir',
        type=str,
        default='results/analysis',
        help='Directory to save analysis outputs (default: results/analysis)'
    )

    args = parser.parse_args()

    # Validate input file
    if not Path(args.results_file).exists():
        print(f"Error: Results file not found: {args.results_file}")
        sys.exit(1)

    print("="*80)
    print("EXPERIMENTAL RESULTS ANALYSIS")
    print("="*80)
    print(f"\nLoading results from: {args.results_file}")

    # Load results
    try:
        results = load_trials(args.results_file)
        print(f"✓ Loaded {len(results)} trial results")
    except Exception as e:
        print(f"Error loading results: {e}")
        sys.exit(1)

    # Create output directory
    output_path = Path(args.output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    print(f"✓ Output directory: {output_path}")

    # Generate comparative plots
    print("\n" + "-"*80)
    print("GENERATING COMPARATIVE PLOTS")
    print("-"*80)
    try:
        plot_all_comparative(results, str(output_path))
    except Exception as e:
        print(f"Error generating plots: {e}")
        import traceback
        traceback.print_exc()

    # Generate statistical report
    print("\n" + "-"*80)
    print("GENERATING STATISTICAL REPORT")
    print("-"*80)
    report_path = output_path / "statistical_report.txt"
    try:
        generate_statistical_report(results, str(report_path))
    except Exception as e:
        print(f"Error generating statistical report: {e}")
        import traceback
        traceback.print_exc()

    print("\n" + "="*80)
    print("ANALYSIS COMPLETE!")
    print("="*80)
    print(f"\nResults saved to: {output_path}")
    print("\nGenerated files:")
    print("  - success_rates.png")
    print("  - computation_time.png")
    print("  - path_length.png")
    print("  - replans_by_complexity.png")
    print("  - belief_convergence.png")
    print("  - statistical_report.txt")
    print()


if __name__ == '__main__':
    main()
