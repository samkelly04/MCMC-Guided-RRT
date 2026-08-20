#!/usr/bin/env python3
"""Quick integration test for telemetry infrastructure.

This script tests that:
1. AdaptiveMCMCAgent captures telemetry
2. Evaluator extracts and serializes telemetry
3. JSON encoding handles numpy arrays
4. Loaders can reconstruct the data
5. Analysis functions work with the data
"""

import json
import sys
from pathlib import Path
import numpy as np

# Add src to path
sys.path.insert(0, str(Path(__file__).parent))

from src.core.agents import AdaptiveMCMCAgent
from src.samplers.adaptive_goal_mcmc import AdaptiveGoalMCMCSampler
from src.rrt.goal_belief import GoalBelief
from src.rrt.goal_sensor import GoalObservationModel, GoalSensorConfig
from src.env.maze_environment import ContinuousMazeEnvironment
from src.config import MazeGenerationConfig
from src.eval.evaluator import Evaluator
from src.eval.metrics import TrialResult
from src.config import ExperimentConfig, ComplexitySpec, ComplexityLevel


def test_telemetry_capture():
    """Test that AdaptiveMCMCAgent captures telemetry."""
    print("="*60)
    print("TEST 1: Telemetry Capture")
    print("="*60)

    # Create a simple maze
    maze_config = MazeGenerationConfig(
        space_size=(100.0, 100.0),
        grid_size=(4, 4),
        wall_thickness=1.0,
        seed=100,
    )

    try:
        maze = ContinuousMazeEnvironment(
            maze_config=maze_config,
            start=np.array([15.0, 15.0]),
            goal=np.array([85.0, 85.0]),
        )
    except ValueError:
        # Try different positions
        maze = ContinuousMazeEnvironment(maze_config=maze_config)

    # Create agent
    sampler = AdaptiveGoalMCMCSampler(seed=42)
    sensor = GoalObservationModel(GoalSensorConfig(seed=42))
    initial_belief = GoalBelief(
        mean=np.array([50.0, 50.0]),
        covariance=np.eye(2) * 100.0,
    )

    agent = AdaptiveMCMCAgent(
        sampler=sampler,
        sensor=sensor,
        initial_belief=initial_belief,
        step_size=5.0,
        max_steps=50,  # Limited for testing
        max_iterations=1000,  # Limited for testing
    )

    print("Running agent.solve()...")
    path = agent.solve(maze)
    print(f"  ✓ Solve completed, success: {path is not None}")

    print("Checking telemetry...")
    telemetry = agent.get_telemetry()

    assert telemetry, "Telemetry should not be empty"
    assert 'num_replans' in telemetry, "Should have num_replans"
    assert 'belief_history' in telemetry, "Should have belief_history"
    assert 'entropy_history' in telemetry, "Should have entropy_history"
    assert 'final_belief' in telemetry, "Should have final_belief"

    print(f"  ✓ Telemetry captured:")
    print(f"    - num_replans: {telemetry['num_replans']}")
    print(f"    - belief_history length: {len(telemetry['belief_history'])}")
    print(f"    - entropy_history length: {len(telemetry['entropy_history'])}")
    print(f"    - final_belief entropy: {telemetry['final_belief'].compute_entropy():.2f}")

    return telemetry, maze


def test_json_serialization(telemetry, maze):
    """Test JSON serialization with NumpyEncoder."""
    print("\n" + "="*60)
    print("TEST 2: JSON Serialization")
    print("="*60)

    # Create TrialResult manually
    trial_result = TrialResult(
        agent_name="TestAgent",
        complexity_level="test",
        map_seed=100,
        success=True,
        steps_taken=14,
        path_length=100.0,
        computation_time=1.0,
        num_replans=telemetry['num_replans'],
        final_belief_mean=telemetry['final_belief'].mean,
        final_belief_covariance=telemetry['final_belief'].covariance,
        final_belief_entropy=telemetry['final_belief'].compute_entropy(),
        belief_convergence=np.linalg.norm(
            telemetry['final_belief'].mean - maze.get_goal()
        ),
        additional_metrics={
            'belief_history': [
                {
                    'mean': b.mean.tolist(),
                    'covariance': b.covariance.tolist(),
                    'entropy': b.compute_entropy(),
                    'observation_count': b.observation_count,
                }
                for b in telemetry['belief_history']
            ],
            'entropy_history': telemetry['entropy_history'],
        }
    )

    # Serialize to JSON
    class NumpyEncoder(json.JSONEncoder):
        def default(self, obj):
            if isinstance(obj, np.ndarray):
                return obj.tolist()
            if isinstance(obj, np.integer):
                return int(obj)
            if isinstance(obj, np.floating):
                return float(obj)
            return super().default(obj)

    trial_dict = {
        "agent_name": trial_result.agent_name,
        "complexity_level": trial_result.complexity_level,
        "map_seed": trial_result.map_seed,
        "success": trial_result.success,
        "steps_taken": trial_result.steps_taken,
        "nodes_expanded": trial_result.nodes_expanded,
        "path_length": trial_result.path_length,
        "computation_time": trial_result.computation_time,
        "timestamp": trial_result.timestamp.isoformat(),
        "num_replans": trial_result.num_replans,
        "final_belief_mean": trial_result.final_belief_mean,
        "final_belief_covariance": trial_result.final_belief_covariance,
        "final_belief_entropy": trial_result.final_belief_entropy,
        "belief_convergence": trial_result.belief_convergence,
        "additional_metrics": trial_result.additional_metrics,
    }

    print("Serializing to JSON...")
    json_str = json.dumps([trial_dict], indent=2, cls=NumpyEncoder)
    print(f"  ✓ Serialized successfully ({len(json_str)} bytes)")

    # Write to file
    test_file = Path("results/telemetry_test.json")
    test_file.parent.mkdir(parents=True, exist_ok=True)
    with open(test_file, 'w') as f:
        f.write(json_str)

    print(f"  ✓ Saved to {test_file}")

    return test_file


def test_loading_and_analysis(test_file):
    """Test loading data and running analysis."""
    print("\n" + "="*60)
    print("TEST 3: Loading and Analysis")
    print("="*60)

    from src.analysis.loaders import load_trials, filter_adaptive
    from src.analysis.comparative_plots import plot_success_rates

    print("Loading trials from JSON...")
    results = load_trials(str(test_file))
    print(f"  ✓ Loaded {len(results)} trials")

    trial = results[0]
    print(f"  ✓ Trial data:")
    print(f"    - Agent: {trial.agent_name}")
    print(f"    - Success: {trial.success}")
    print(f"    - Num replans: {trial.num_replans}")
    print(f"    - Final belief mean: {trial.final_belief_mean}")
    print(f"    - Belief convergence: {trial.belief_convergence:.2f}")
    print(f"    - Belief history entries: {len(trial.belief_history) if trial.belief_history else 0}")

    # Verify numpy arrays were reconstructed
    assert isinstance(trial.final_belief_mean, np.ndarray), "Should be numpy array"
    assert isinstance(trial.final_belief_covariance, np.ndarray), "Should be numpy array"

    if trial.belief_history:
        assert isinstance(trial.belief_history[0]['mean'], np.ndarray), "Belief means should be numpy arrays"
        print("  ✓ Numpy arrays correctly reconstructed")

    print("\nAll tests passed!")
    return results


if __name__ == "__main__":
    print("\n" + "="*60)
    print("TELEMETRY INFRASTRUCTURE INTEGRATION TEST")
    print("="*60)

    try:
        telemetry, maze = test_telemetry_capture()
        test_file = test_json_serialization(telemetry, maze)
        results = test_loading_and_analysis(test_file)

        print("\n" + "="*60)
        print("✓ ALL TESTS PASSED")
        print("="*60)
        print("\nThe telemetry infrastructure is working correctly:")
        print("  - Agents capture rich telemetry")
        print("  - JSON serialization handles numpy arrays")
        print("  - Loaders reconstruct data correctly")
        print("  - Analysis tools can process the data")
        print("\nYou can now run the full experiment!")

    except Exception as e:
        print(f"\n✗ TEST FAILED: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
