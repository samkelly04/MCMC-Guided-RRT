#!/usr/bin/env python3
"""Batch comparison: Receding-Horizon Adaptive MCMC vs Single-Shot Uniform RRT.

This experiment compares two path planning strategies:

1. RRTAgent (Uniform Sampling)
   - Single-shot RRT with uniform random sampling
   - Perfect goal knowledge
   - Baseline comparison
   
2. AdaptiveMCMCAgent (Receding-Horizon with Active Perception)
   - Receding-horizon navigation with replanning
   - Initial goal uncertainty (belief centered at workspace middle)
   - Goal belief refines via sensor observations as robot approaches
   - MCMC sampling with information gain term for active perception
   - Event-triggered replanning on significant entropy reduction
   - 30% uniform mixing rate for exploration

Key parameters:
- MCMC: lambda_clearance=0 (removed), uniform_mixing_rate=0.30
- Initial goal belief: N([50,50], 400*I) - uncertain about true goal
- Sensor: far_std=10, near_std=0.5 - accuracy improves with proximity

Runs across 3 complexity levels (Easy/Medium/Hard) with configurable seeds.
"""

import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import numpy as np

from src.config import (
    ExperimentConfig,
    ComplexityLevel,
    ComplexitySpec,
    MazeGenerationConfig,
)
from src.eval import Evaluator
from src.core.agents import RRTAgent, AdaptiveMCMCAgent
from src.samplers.uniform import UniformSampler
from src.samplers.adaptive_goal_mcmc import AdaptiveGoalMCMCSampler
from src.rrt.goal_belief import GoalBelief
from src.rrt.goal_sensor import GoalObservationModel, GoalSensorConfig


def create_uniform_agent(seed: int = 42) -> RRTAgent:
    """Create a uniform sampling RRT agent."""
    sampler = UniformSampler(seed=seed)
    agent = RRTAgent(
        sampler=sampler,
        step_size=5.0,  # Larger step for faster tree expansion
        goal_threshold=5.0,
        max_iterations=15000,  # More iterations for complex mazes
    )
    return agent


def create_adaptive_mcmc_agent(
    seed: int = 42,
    initial_uncertainty: float = 400.0,
) -> AdaptiveMCMCAgent:
    """Create an adaptive MCMC RRT agent with receding-horizon navigation.
    
    This agent uses the full receding-horizon navigation with event-triggered
    replanning. Key features:
    - Goal belief starts uncertain (centered at workspace middle)
    - Sensor observations refine goal estimate as robot approaches
    - Active perception via information gain term
    - Replanning triggered by significant entropy reduction
    
    Args:
        seed: Random seed for reproducibility.
        initial_uncertainty: Initial variance for goal belief (higher = more uncertain).
    """
    sampler = AdaptiveGoalMCMCSampler(
        proposal_std=8.0,
        temperature=10.0,
        lambda_goal=1.0,
        lambda_clearance=0.0,  # REMOVED - standard RRT doesn't have this
        lambda_info=2.0,       # Active perception weight
        uniform_mixing_rate=0.30,  # 30% uniform for exploration
        seed=seed,
    )
    
    sensor = GoalObservationModel(
        GoalSensorConfig(
            far_std=10.0,   # High uncertainty when far
            near_std=0.5,   # Low uncertainty when close
            distance_scale=50.0,
            seed=seed,
        )
    )
    
    # Initial belief: uncertain about goal location
    # Centered at workspace middle with high uncertainty
    initial_belief = GoalBelief(
        mean=np.array([50.0, 50.0]),
        covariance=np.eye(2) * initial_uncertainty,
    )
    
    agent = AdaptiveMCMCAgent(
        sampler=sampler,
        sensor=sensor,
        initial_belief=initial_belief,
        step_size=5.0,
        goal_threshold=5.0,
        goal_tolerance=8.0,
        max_steps=500,        # Increased from 300 - allow more exploration for harder mazes
        execution_horizon=10, # Execute 10 waypoints before considering replan
        entropy_threshold=5.0, # Trigger replan on significant info gain
        max_iterations=15000, # Increased from 10000 - match RRTAgent's capacity
        min_steps_between_replans=3,
    )
    
    return agent


def create_experiment_config(
    num_maps: int = 30,
    trials_per_map: int = 1,
    output_dir: str = "results/batch_comparison",
) -> ExperimentConfig:
    """Create experiment configuration with three complexity levels.
    
    Args:
        num_maps: Number of unique maps per complexity level.
        trials_per_map: Number of trials per map (for stochastic agents).
        output_dir: Directory for saving results.
    
    Returns:
        ExperimentConfig with Easy (10x10), Medium (15x15), Hard (20x20) levels.
    """
    # Easy: 6x6 grid (sparse walls, fast to solve)
    easy_config = MazeGenerationConfig(
        space_size=(100.0, 100.0),
        grid_size=(6, 6),
        wall_thickness=1.0,
    )
    
    easy_spec = ComplexitySpec(
        level=ComplexityLevel.EASY,
        maze_config=easy_config,
        num_maps=num_maps,
        trials_per_map=trials_per_map,
    )
    
    # Medium: 8x8 grid (moderate complexity)
    medium_config = MazeGenerationConfig(
        space_size=(100.0, 100.0),
        grid_size=(8, 8),
        wall_thickness=1.0,
    )
    
    medium_spec = ComplexitySpec(
        level=ComplexityLevel.MEDIUM,
        maze_config=medium_config,
        num_maps=num_maps,
        trials_per_map=trials_per_map,
    )
    
    # Hard: 10x10 grid (challenging but solvable)
    hard_config = MazeGenerationConfig(
        space_size=(100.0, 100.0),
        grid_size=(10, 10),
        wall_thickness=1.0,
    )
    
    hard_spec = ComplexitySpec(
        level=ComplexityLevel.HARD,
        maze_config=hard_config,
        num_maps=num_maps,
        trials_per_map=trials_per_map,
    )
    
    config = ExperimentConfig(
        complexity_levels=[easy_spec, medium_spec, hard_spec],
        output_dir=output_dir,
        save_individual_results=True,
        save_aggregated_results=True,
        verbose=True,
    )
    
    return config


def run_experiment(
    num_maps: int = 30,
    trials_per_map: int = 1,
    seed_offset: int = 100,
    output_dir: str = "results/batch_comparison",
):
    """Run the batch comparison experiment.
    
    Args:
        num_maps: Number of unique maps per complexity level (default: 30).
        trials_per_map: Number of trials per map (default: 1).
        seed_offset: Starting seed for maze generation (default: 100).
        output_dir: Directory for saving results.
    """
    print("="*70)
    print("BATCH COMPARISON: Adaptive MCMC vs Uniform Sampling")
    print("="*70)
    print(f"\nConfiguration:")
    print(f"  Maps per complexity level: {num_maps}")
    print(f"  Trials per map: {trials_per_map}")
    print(f"  Seed range: {seed_offset} - {seed_offset + num_maps * 3 - 1}")
    print(f"  Total trials per agent: {num_maps * trials_per_map * 3}")
    print(f"  Output directory: {output_dir}")
    
    # Create experiment config
    config = create_experiment_config(
        num_maps=num_maps,
        trials_per_map=trials_per_map,
        output_dir=output_dir,
    )
    
    # Create agents
    uniform_agent = create_uniform_agent(seed=42)
    adaptive_agent = create_adaptive_mcmc_agent(seed=42, initial_uncertainty=400.0)
    
    agents = [uniform_agent, adaptive_agent]
    
    print(f"\nAgents:")
    for agent in agents:
        print(f"  - {agent.get_name()}")
    
    # Create evaluator and run
    evaluator = Evaluator(config)
    evaluator.run_experiment(agents, seed_offset=seed_offset)
    
    # Save results
    evaluator.save_results()
    
    # Print summary
    evaluator.print_summary()
    
    return evaluator


def run_quick_test():
    """Run a quick test with fewer maps to verify everything works."""
    print("\n" + "="*70)
    print("QUICK TEST (3 maps per level, 1 trial each)")
    print("="*70)
    
    return run_experiment(
        num_maps=3,
        trials_per_map=1,
        seed_offset=100,
        output_dir="results/quick_test",
    )


def run_full_experiment():
    """Run the full experiment as specified: 30 maps per level, seeds 100-129."""
    print("\n" + "="*70)
    print("FULL EXPERIMENT (30 maps per level)")
    print("="*70)
    
    return run_experiment(
        num_maps=30,
        trials_per_map=1,
        seed_offset=100,
        output_dir="results/full_comparison",
    )


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Run batch comparison of Adaptive MCMC vs Uniform sampling"
    )
    parser.add_argument(
        "--quick", 
        action="store_true",
        help="Run quick test with 3 maps per level instead of 30"
    )
    parser.add_argument(
        "--num-maps",
        type=int,
        default=30,
        help="Number of maps per complexity level (default: 30)"
    )
    parser.add_argument(
        "--trials",
        type=int,
        default=1,
        help="Number of trials per map (default: 1)"
    )
    parser.add_argument(
        "--seed-offset",
        type=int,
        default=100,
        help="Starting seed for maze generation (default: 100)"
    )
    parser.add_argument(
        "--output",
        type=str,
        default="results/batch_comparison",
        help="Output directory for results"
    )
    
    args = parser.parse_args()
    
    if args.quick:
        evaluator = run_quick_test()
    else:
        evaluator = run_experiment(
            num_maps=args.num_maps,
            trials_per_map=args.trials,
            seed_offset=args.seed_offset,
            output_dir=args.output,
        )
    
    output_dir = args.output if not args.quick else "results/quick_test"
    print(f"\n✓ Experiment complete! Results saved to: {output_dir}")

    # Print analysis instructions
    print("\n" + "="*70)
    print("NEXT STEPS: Analyze Your Results")
    print("="*70)
    print("\n1. Generate comparative plots and statistical analysis:")
    print(f"   python experiments/analyze_results.py {output_dir}/trials_*.json")
    print("   → Creates: success_rates.png, computation_time.png, path_length.png,")
    print("              replans_by_complexity.png, belief_convergence.png,")
    print("              statistical_report.txt")

    print("\n2. Visualize belief evolution for individual trials:")
    print(f"   python experiments/visualize_belief_evolution.py {output_dir}/trials_*.json --trial-index 0")
    print("   → Creates: belief evolution plots with uncertainty ellipses")
    print("   Options: --agent AdaptiveMCMCAgent --complexity easy --plot-all")

    print("\n3. Analyze replanning behavior:")
    print(f"   python experiments/analyze_replanning.py {output_dir}/trials_*.json")
    print("   → Creates: replan_frequency_histogram.png, replans_vs_success.png,")
    print("              entropy_reduction_analysis.png, replan_analysis_report.txt")

    print("\n" + "="*70)

