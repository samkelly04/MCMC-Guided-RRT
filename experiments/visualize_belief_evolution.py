#!/usr/bin/env python3
"""Script to visualize belief evolution for individual trials.

This script generates belief evolution plots and entropy plots for specific
trials from adaptive agents.

Usage:
    python experiments/visualize_belief_evolution.py results/batch_comparison/trials_*.json --trial-index 0
    python experiments/visualize_belief_evolution.py results/batch_comparison/trials_*.json --agent AdaptiveMCMCAgent --complexity easy
    python experiments/visualize_belief_evolution.py results/batch_comparison/trials_*.json --trial-index 5 --output belief_trial_5.png
"""

import argparse
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.analysis.loaders import (
    load_trials,
    filter_by_agent,
    filter_by_complexity,
    filter_adaptive,
)
from src.analysis.belief_visualization import (
    plot_belief_evolution_from_trial,
    plot_entropy_evolution_from_trial,
    plot_belief_convergence_over_time,
)


def main():
    parser = argparse.ArgumentParser(
        description='Visualize belief evolution for individual trials.'
    )
    parser.add_argument(
        'results_file',
        type=str,
        help='Path to JSON file containing trial results'
    )
    parser.add_argument(
        '--trial-index',
        type=int,
        default=None,
        help='Index of trial to visualize (after filtering)'
    )
    parser.add_argument(
        '--agent',
        type=str,
        default=None,
        help='Filter to specific agent name'
    )
    parser.add_argument(
        '--complexity',
        type=str,
        default=None,
        help='Filter to specific complexity level (easy, medium, hard)'
    )
    parser.add_argument(
        '--successful-only',
        action='store_true',
        help='Only show successful trials'
    )
    parser.add_argument(
        '--output',
        type=str,
        default=None,
        help='Output file path (default: displays plot interactively)'
    )
    parser.add_argument(
        '--output-dir',
        type=str,
        default='results/belief_viz',
        help='Directory to save multiple plots (default: results/belief_viz)'
    )
    parser.add_argument(
        '--plot-all',
        action='store_true',
        help='Generate plots for all matching trials'
    )
    parser.add_argument(
        '--plot-type',
        type=str,
        choices=['belief', 'entropy', 'convergence', 'all'],
        default='belief',
        help='Type of plot to generate (default: belief)'
    )

    args = parser.parse_args()

    # Validate input file
    if not Path(args.results_file).exists():
        print(f"Error: Results file not found: {args.results_file}")
        sys.exit(1)

    print("="*80)
    print("BELIEF EVOLUTION VISUALIZATION")
    print("="*80)
    print(f"\nLoading results from: {args.results_file}")

    # Load results
    try:
        results = load_trials(args.results_file)
        print(f"✓ Loaded {len(results)} trial results")
    except Exception as e:
        print(f"Error loading results: {e}")
        sys.exit(1)

    # Filter to adaptive agents only
    results = filter_adaptive(results)
    print(f"✓ Found {len(results)} adaptive agent trials")

    if not results:
        print("Error: No adaptive agent trials found")
        sys.exit(1)

    # Apply filters
    if args.agent:
        results = filter_by_agent(results, args.agent)
        print(f"✓ Filtered to agent '{args.agent}': {len(results)} trials")

    if args.complexity:
        results = filter_by_complexity(results, args.complexity)
        print(f"✓ Filtered to '{args.complexity}' complexity: {len(results)} trials")

    if args.successful_only:
        results = [r for r in results if r.success]
        print(f"✓ Filtered to successful trials: {len(results)} trials")

    if not results:
        print("Error: No trials match the specified filters")
        sys.exit(1)

    # Determine which trials to visualize
    if args.plot_all:
        trial_indices = list(range(len(results)))
        output_dir = Path(args.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        print(f"\n✓ Will generate plots for all {len(results)} trials")
        print(f"✓ Output directory: {output_dir}")
    else:
        if args.trial_index is None:
            args.trial_index = 0
        if args.trial_index >= len(results):
            print(f"Error: Trial index {args.trial_index} out of range (0-{len(results)-1})")
            sys.exit(1)
        trial_indices = [args.trial_index]
        print(f"\n✓ Will visualize trial {args.trial_index}")

    # Generate plots
    print("\n" + "-"*80)
    print("GENERATING PLOTS")
    print("-"*80)

    for idx in trial_indices:
        trial = results[idx]

        # Prepare output paths
        if args.plot_all:
            base_name = f"trial_{idx}_{trial.agent_name}_{trial.complexity_level}_seed{trial.map_seed}"
            belief_path = str(output_dir / f"{base_name}_belief.png")
            entropy_path = str(output_dir / f"{base_name}_entropy.png")
            convergence_path = str(output_dir / f"{base_name}_convergence.png")
        else:
            if args.output:
                # Use user-specified output
                output_path = Path(args.output)
                belief_path = str(output_path.with_suffix('.png'))
                entropy_path = str(output_path.with_stem(output_path.stem + '_entropy').with_suffix('.png'))
                convergence_path = str(output_path.with_stem(output_path.stem + '_convergence').with_suffix('.png'))
            else:
                # Display interactively
                belief_path = None
                entropy_path = None
                convergence_path = None

        # Reconstruct true goal from belief convergence
        # Since we don't store the true goal, we approximate it using final belief + convergence
        if trial.final_belief_mean is not None and trial.belief_convergence is not None:
            # This is an approximation - we don't have the exact direction
            # For visualization purposes, use final belief mean as proxy
            true_goal = trial.final_belief_mean.copy()
            print(f"\nNote: Using final belief mean as true goal proxy for trial {idx}")
        else:
            print(f"\nWarning: Trial {idx} missing belief data, skipping...")
            continue

        # Generate requested plot types
        if args.plot_type in ['belief', 'all']:
            try:
                plot_belief_evolution_from_trial(trial, true_goal, save_path=belief_path)
                if belief_path:
                    print(f"  ✓ Saved belief evolution: {belief_path}")
                else:
                    print(f"  ✓ Displayed belief evolution for trial {idx}")
            except Exception as e:
                print(f"  ✗ Error generating belief plot: {e}")

        if args.plot_type in ['entropy', 'all']:
            try:
                plot_entropy_evolution_from_trial(trial, save_path=entropy_path)
                if entropy_path:
                    print(f"  ✓ Saved entropy evolution: {entropy_path}")
                else:
                    print(f"  ✓ Displayed entropy evolution for trial {idx}")
            except Exception as e:
                print(f"  ✗ Error generating entropy plot: {e}")

        if args.plot_type in ['convergence', 'all']:
            try:
                if trial.belief_history:
                    plot_belief_convergence_over_time(
                        trial.belief_history,
                        true_goal,
                        save_path=convergence_path
                    )
                    if convergence_path:
                        print(f"  ✓ Saved belief convergence: {convergence_path}")
                    else:
                        print(f"  ✓ Displayed belief convergence for trial {idx}")
            except Exception as e:
                print(f"  ✗ Error generating convergence plot: {e}")

    print("\n" + "="*80)
    print("VISUALIZATION COMPLETE!")
    print("="*80)


if __name__ == '__main__':
    main()
