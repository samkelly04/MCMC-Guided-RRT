"""Example usage of the evaluation framework (scaffold demonstration).

This file demonstrates how to set up and configure the evaluation framework.
The actual execution loops are not implemented here - this shows the structure.
"""

from src.config import (
    ExperimentConfig,
    ComplexityLevel,
    ComplexitySpec,
    MazeGenerationConfig,
)
from src.eval import Evaluator


def create_example_config() -> ExperimentConfig:
    """Create an example experiment configuration.
    
    This demonstrates how to set up a comprehensive evaluation experiment
    with multiple complexity levels, different numbers of maps, and multiple
    trials per map for stochastic agents.
    """
    # Define complexity levels with different maze configurations
    
    # Easy: Small grid, sparse walls
    easy_config = MazeGenerationConfig(
        space_size=(50.0, 50.0),
        grid_size=(10, 10),
        wall_thickness=1.0,
    )
    
    easy_spec = ComplexitySpec(
        level=ComplexityLevel.EASY,
        maze_config=easy_config,
        num_maps=20,  # Generate 20 unique easy mazes
        trials_per_map=1,  # Run once per map (deterministic agent)
    )
    
    # Medium: Medium grid, moderate complexity
    medium_config = MazeGenerationConfig(
        space_size=(100.0, 100.0),
        grid_size=(15, 15),
        wall_thickness=1.0,
    )
    
    medium_spec = ComplexitySpec(
        level=ComplexityLevel.MEDIUM,
        maze_config=medium_config,
        num_maps=50,  # Generate 50 unique medium mazes
        trials_per_map=5,  # Run 5 times per map (stochastic agent)
    )
    
    # Hard: Large grid, dense walls
    hard_config = MazeGenerationConfig(
        space_size=(100.0, 100.0),
        grid_size=(20, 20),
        wall_thickness=1.0,
    )
    
    hard_spec = ComplexitySpec(
        level=ComplexityLevel.HARD,
        maze_config=hard_config,
        num_maps=50,  # Generate 50 unique hard mazes
        trials_per_map=10,  # Run 10 times per map (highly stochastic agent)
    )
    
    # Create experiment configuration
    config = ExperimentConfig(
        complexity_levels=[easy_spec, medium_spec, hard_spec],
        output_dir="results/experiment_001",
        save_individual_results=True,
        save_aggregated_results=True,
        verbose=True,
    )
    
    return config


def example_usage():
    """Example of how to use the evaluation framework."""
    
    # Create configuration
    config = create_example_config()
    
    print(f"Total maps to generate: {config.get_total_maps()}")
    print(f"Total trials to run: {config.get_total_trials()}")
    
    # Create evaluator
    evaluator = Evaluator(config)
    
    # Note: The following would require actual agent implementations
    # agents = [MyRRTAgent(), MyAStarAgent(), MyRLAgent()]
    # results = evaluator.run_experiment(agents)
    # evaluator.save_results()
    # evaluator.print_summary()
    
    print("\n✓ Evaluation framework scaffold is ready!")
    print("  Next steps:")
    print("  1. Implement MazeEnvironment wrapper for your maze generator")
    print("  2. Implement MazeAgent wrapper for your pathfinding algorithms")
    print("  3. Implement generate_mazes() in Evaluator")
    print("  4. Implement run_trial() in Evaluator")
    print("  5. Implement run_experiment() loop in Evaluator")


if __name__ == "__main__":
    example_usage()
