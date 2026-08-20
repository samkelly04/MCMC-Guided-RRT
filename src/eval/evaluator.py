"""Main evaluator class for running maze-solving experiments."""

import json
import pickle
import time
from collections import Counter
from pathlib import Path
from typing import List, Optional
from dataclasses import asdict
from datetime import datetime

import numpy as np

from ..core.agent import MazeAgent
from ..core.environment import MazeEnvironment
from ..config.experiment_config import ExperimentConfig, ComplexitySpec, MazeGenerationConfig
from ..env.maze_environment import ContinuousMazeEnvironment
from .metrics import MetricTracker, TrialResult


class Evaluator:
    """Main evaluator class for running comprehensive maze-solving experiments.
    
    This class orchestrates the evaluation process:
    1. Generates mazes according to complexity specifications
    2. Runs agents on each maze
    3. Collects and aggregates metrics
    4. Saves results to disk
    """
    
    def __init__(self, config: ExperimentConfig):
        """Initialize the evaluator with experiment configuration.
        
        Args:
            config: ExperimentConfig defining the experiment parameters.
        """
        self.config = config
        self.metric_tracker = MetricTracker()
        self._ensure_output_dir()
    
    def _ensure_output_dir(self) -> None:
        """Create output directory if it doesn't exist."""
        output_path = Path(self.config.output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
    
    def setup_experiment(self) -> None:
        """Set up the experiment environment."""
        if self.config.verbose:
            print(f"\n{'='*60}")
            print(f"Experiment Setup")
            print(f"{'='*60}")
            print(f"Total trials to run: {self.config.get_total_trials()}")
            print(f"Total unique maps: {self.config.get_total_maps()}")
            print(f"Output directory: {self.config.output_dir}")
            print(f"{'='*60}\n")
    
    def generate_mazes(
        self, 
        complexity_spec: ComplexitySpec,
        start_seed: int = 0,
    ) -> List[ContinuousMazeEnvironment]:
        """Generate mazes for a given complexity specification.
        
        Args:
            complexity_spec: ComplexitySpec defining maze generation parameters.
            start_seed: Starting seed number for maze generation.
        
        Returns:
            List of ContinuousMazeEnvironment instances, one for each unique map seed.
        """
        if self.config.verbose:
            print(f"Generating {complexity_spec.num_maps} mazes for "
                  f"{complexity_spec.level.value} complexity (seeds {start_seed}-{start_seed + complexity_spec.num_maps - 1})")
        
        mazes = []
        base_config = complexity_spec.maze_config
        
        for i in range(complexity_spec.num_maps):
            seed = start_seed + i
            
            # Create config with this seed
            maze_config = MazeGenerationConfig(
                space_size=base_config.space_size,
                grid_size=base_config.grid_size,
                wall_thickness=base_config.wall_thickness,
                seed=seed,
            )
            
            # Generate maze with fixed start/goal positions
            # Use corner positions for consistency across experiments
            space_w, space_h = maze_config.space_size
            margin = max(space_w, space_h) * 0.12  # 12% margin from edges
            
            # Try multiple corner configurations if the first fails
            corner_configs = [
                (np.array([margin, margin]), np.array([space_w - margin, space_h - margin])),
                (np.array([margin, space_h - margin]), np.array([space_w - margin, margin])),
                (np.array([space_w / 4, space_h / 4]), np.array([3 * space_w / 4, 3 * space_h / 4])),
            ]
            
            maze = None
            for start_pos, goal_pos in corner_configs:
                try:
                    maze = ContinuousMazeEnvironment(
                        maze_config=maze_config,
                        start=start_pos,
                        goal=goal_pos,
                    )
                    break
                except ValueError:
                    continue
            
            if maze is None:
                # Fall back to auto-generated positions
                if self.config.verbose:
                    print(f"  Warning: seed {seed} - using auto-generated start/goal")
                maze = ContinuousMazeEnvironment(maze_config=maze_config)
            
            mazes.append(maze)
        
        return mazes
    
    def run_trial(
        self,
        agent: MazeAgent,
        environment: MazeEnvironment,
        complexity_level: str,
        map_seed: Optional[int] = None,
        trial_seed: Optional[int] = None,
    ) -> TrialResult:
        """Run a single trial of an agent on a maze.
        
        Args:
            agent: MazeAgent to evaluate.
            environment: MazeEnvironment to solve.
            complexity_level: String identifier for complexity level.
            map_seed: Seed used to generate this maze.
            trial_seed: Seed for agent randomness (for stochastic agents).
        
        Returns:
            TrialResult containing all metrics from this trial.
        """
        # Reset agent with trial seed
        if trial_seed is not None:
            agent.reset(trial_seed)
        else:
            agent.reset(map_seed)
        
        # Time the solve
        start_time = time.perf_counter()
        path = agent.solve(environment)
        end_time = time.perf_counter()

        computation_time = end_time - start_time
        success = path is not None and len(path) > 0

        # Compute metrics
        steps_taken = len(path) if path else 0
        path_length = self._compute_path_length(path) if path else 0.0

        # Check if we actually reached the goal
        if success:
            goal = environment.get_goal()
            final_pos = path[-1]
            distance_to_goal = float(np.linalg.norm(final_pos - goal))
            # Consider success if within reasonable distance
            if distance_to_goal > 10.0:  # Allow some tolerance
                success = False

        # Extract telemetry from agent (for adaptive agents)
        telemetry = agent.get_telemetry()

        # Process adaptive navigation metrics if available
        num_replans = None
        final_belief_mean = None
        final_belief_covariance = None
        final_belief_entropy = None
        belief_convergence = None
        additional_metrics = {"trial_seed": trial_seed}

        if telemetry:
            # Extract basic metrics
            num_replans = telemetry.get('num_replans')

            # Process final belief
            final_belief = telemetry.get('final_belief')
            if final_belief is not None:
                final_belief_mean = final_belief.mean.copy()
                final_belief_covariance = final_belief.covariance.copy()
                final_belief_entropy = final_belief.compute_entropy()

                # Compute belief convergence (distance from belief to true goal)
                goal = environment.get_goal()
                belief_convergence = float(np.linalg.norm(final_belief.mean - goal))

            # Serialize belief and entropy histories for storage
            belief_history = telemetry.get('belief_history', [])
            entropy_history = telemetry.get('entropy_history', [])

            if belief_history:
                additional_metrics['belief_history'] = [
                    {
                        'mean': b.mean.tolist(),
                        'covariance': b.covariance.tolist(),
                        'entropy': b.compute_entropy(),
                        'observation_count': b.observation_count,
                    }
                    for b in belief_history
                ]

            if entropy_history:
                additional_metrics['entropy_history'] = entropy_history

            # Store other telemetry items
            if 'num_steps' in telemetry:
                additional_metrics['num_steps'] = telemetry['num_steps']
            if 'distance_to_goal' in telemetry:
                additional_metrics['distance_to_goal'] = telemetry['distance_to_goal']

        # Extract MCMC sample data for C-space learning
        sampler_log = []
        if hasattr(agent, "sampler") and hasattr(agent.sampler, "get_sample_log"):
            sampler_log = agent.sampler.get_sample_log()
            if sampler_log:
                type_counts = Counter(entry[1] for entry in sampler_log)
                additional_metrics["sample_type_counts"] = dict(type_counts)
                additional_metrics["total_samples"] = len(sampler_log)

        segment_log = []
        if hasattr(agent, "_planner") and agent._planner is not None:
            segment_log = getattr(agent._planner, "segment_log", [])
            if segment_log:
                additional_metrics["total_segments"] = len(segment_log)
                additional_metrics["segment_collision_count"] = sum(
                    1 for _, _, free in segment_log if not free
                )

        # Store logs temporarily for save_sample_data() to access
        self._last_sampler_log = sampler_log
        self._last_segment_log = segment_log

        return TrialResult(
            agent_name=agent.get_name(),
            complexity_level=complexity_level,
            map_seed=map_seed,
            success=success,
            steps_taken=steps_taken,
            nodes_expanded=0,  # Could be tracked by agent if needed
            path_length=path_length,
            computation_time=computation_time,
            path=path if self.config.save_individual_results else None,
            num_replans=num_replans,
            final_belief_mean=final_belief_mean,
            final_belief_covariance=final_belief_covariance,
            final_belief_entropy=final_belief_entropy,
            belief_convergence=belief_convergence,
            additional_metrics=additional_metrics,
        )
    
    def _compute_path_length(self, path: List[np.ndarray]) -> float:
        """Compute total Euclidean length of a path."""
        if not path or len(path) < 2:
            return 0.0
        
        total_length = 0.0
        for i in range(len(path) - 1):
            segment_length = np.linalg.norm(path[i + 1] - path[i])
            total_length += segment_length
        
        return float(total_length)
    
    def record_result(self, result: TrialResult) -> None:
        """Record a trial result in the metric tracker.

        Args:
            result: TrialResult to record.
        """
        self.metric_tracker.record_result(result)

    def save_sample_data(
        self,
        result: TrialResult,
        trial_index: int,
    ) -> None:
        """Save MCMC sample data from the last trial as a pickle file.

        Saves sampler and segment logs collected during the most recent
        run_trial() call. Call this immediately after run_trial().

        Args:
            result: The TrialResult from the trial.
            trial_index: Sequential trial index for filename.
        """
        sampler_log = getattr(self, "_last_sampler_log", [])
        segment_log = getattr(self, "_last_segment_log", [])

        if not sampler_log and not segment_log:
            return

        sample_dir = Path(self.config.output_dir) / "sample_data"
        sample_dir.mkdir(parents=True, exist_ok=True)

        data = {
            "trial_index": trial_index,
            "agent_name": result.agent_name,
            "complexity_level": result.complexity_level,
            "map_seed": result.map_seed,
            "success": result.success,
            "sampler_log": sampler_log,
            "segment_log": segment_log,
            "num_sampler_entries": len(sampler_log),
            "num_segment_entries": len(segment_log),
        }

        filepath = sample_dir / f"samples_trial_{trial_index:04d}.pkl"
        with open(filepath, "wb") as f:
            pickle.dump(data, f, protocol=pickle.HIGHEST_PROTOCOL)
    
    def run_experiment(
        self, 
        agents: List[MazeAgent],
        seed_offset: int = 100,
    ) -> MetricTracker:
        """Run the complete experiment across all agents and complexity levels.
        
        Args:
            agents: List of MazeAgent instances to evaluate.
            seed_offset: Starting seed for maze generation (default: 100).
        
        Returns:
            MetricTracker containing all trial results.
        """
        if self.config.verbose:
            print(f"\n{'='*60}")
            print(f"Starting Experiment")
            print(f"{'='*60}")
            print(f"Agents: {[a.get_name() for a in agents]}")
            print(f"Complexity levels: {[spec.level.value for spec in self.config.complexity_levels]}")
            print(f"Seed offset: {seed_offset}")
        
        self.setup_experiment()
        
        total_trials = self.config.get_total_trials() * len(agents)
        completed_trials = 0
        
        current_seed_offset = seed_offset
        
        for complexity_spec in self.config.complexity_levels:
            level_name = complexity_spec.level.value
            
            if self.config.verbose:
                print(f"\n--- {level_name.upper()} Complexity ---")
                print(f"Grid size: {complexity_spec.maze_config.grid_size}")
                print(f"Maps: {complexity_spec.num_maps}, Trials per map: {complexity_spec.trials_per_map}")
            
            # Generate mazes for this complexity level
            mazes = self.generate_mazes(complexity_spec, start_seed=current_seed_offset)
            
            for agent in agents:
                agent_name = agent.get_name()
                
                if self.config.verbose:
                    print(f"\n  Agent: {agent_name}")
                
                successes = 0
                total_time = 0.0
                
                for maze_idx, maze in enumerate(mazes):
                    map_seed = current_seed_offset + maze_idx

                    for trial in range(complexity_spec.trials_per_map):
                        # Use different trial seed for each trial on same map
                        trial_seed = map_seed * 1000 + trial

                        # Clear sampler log before each trial
                        if hasattr(agent, "sampler") and hasattr(agent.sampler, "clear_sample_log"):
                            agent.sampler.clear_sample_log()

                        result = self.run_trial(
                            agent=agent,
                            environment=maze,
                            complexity_level=level_name,
                            map_seed=map_seed,
                            trial_seed=trial_seed,
                        )

                        self.record_result(result)
                        self.save_sample_data(result, completed_trials)
                        completed_trials += 1

                        if result.success:
                            successes += 1
                        total_time += result.computation_time

                        # Progress indicator
                        if self.config.verbose and (completed_trials % 10 == 0 or completed_trials == total_trials):
                            pct = (completed_trials / total_trials) * 100
                            print(f"\r    Progress: {completed_trials}/{total_trials} ({pct:.1f}%)", end="")
                
                # Print agent summary for this complexity level
                num_trials = complexity_spec.num_maps * complexity_spec.trials_per_map
                success_rate = successes / num_trials if num_trials > 0 else 0
                avg_time = total_time / num_trials if num_trials > 0 else 0
                
                if self.config.verbose:
                    print(f"\n    {agent_name}: {successes}/{num_trials} success ({success_rate:.1%}), "
                          f"avg time: {avg_time:.3f}s")
            
            # Move seed offset for next complexity level
            current_seed_offset += complexity_spec.num_maps
        
        if self.config.verbose:
            print(f"\n{'='*60}")
            print(f"Experiment Complete!")
            print(f"{'='*60}")
        
        return self.metric_tracker
    
    def save_results(self, filepath: Optional[str] = None) -> None:
        """Save experiment results to disk.

        Args:
            filepath: Optional custom filepath. If None, uses config.output_dir.
        """
        # Custom JSON encoder for numpy types
        class NumpyEncoder(json.JSONEncoder):
            def default(self, obj):
                if isinstance(obj, np.ndarray):
                    return obj.tolist()
                if isinstance(obj, np.integer):
                    return int(obj)
                if isinstance(obj, np.floating):
                    return float(obj)
                return super().default(obj)

        output_path = Path(filepath) if filepath else Path(self.config.output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Save individual trial results as JSON
        if self.config.save_individual_results:
            trials_file = output_path / f"trials_{timestamp}.json"
            trials_data = []

            for result in self.metric_tracker.results:
                trial_dict = {
                    "agent_name": result.agent_name,
                    "complexity_level": result.complexity_level,
                    "map_seed": result.map_seed,
                    "success": result.success,
                    "steps_taken": result.steps_taken,
                    "nodes_expanded": result.nodes_expanded,
                    "path_length": result.path_length,
                    "computation_time": result.computation_time,
                    "timestamp": result.timestamp.isoformat(),
                    # Adaptive navigation metrics
                    "num_replans": result.num_replans,
                    "final_belief_mean": result.final_belief_mean,
                    "final_belief_covariance": result.final_belief_covariance,
                    "final_belief_entropy": result.final_belief_entropy,
                    "belief_convergence": result.belief_convergence,
                    "additional_metrics": result.additional_metrics,
                }
                trials_data.append(trial_dict)

            with open(trials_file, "w") as f:
                json.dump(trials_data, f, indent=2, cls=NumpyEncoder)

            if self.config.verbose:
                print(f"Saved individual trials to: {trials_file}")
        
        # Save aggregated statistics
        if self.config.save_aggregated_results:
            stats_file = output_path / f"stats_{timestamp}.json"
            
            # Get unique agent names and complexity levels
            agent_names = list(set(r.agent_name for r in self.metric_tracker.results))
            complexity_levels = list(set(r.complexity_level for r in self.metric_tracker.results))
            
            stats_data = {
                "experiment_timestamp": timestamp,
                "total_trials": len(self.metric_tracker.results),
                "agents": agent_names,
                "complexity_levels": complexity_levels,
                "overall": self.metric_tracker.get_statistics_summary(),
                "by_complexity": {},
                "by_agent": {},
                "by_agent_complexity": {},
            }
            
            # Stats by complexity level
            for level in complexity_levels:
                stats_data["by_complexity"][level] = self.metric_tracker.get_statistics_summary(level)
            
            # Stats by agent
            for agent_name in agent_names:
                agent_results = [r for r in self.metric_tracker.results if r.agent_name == agent_name]
                agent_tracker = MetricTracker(results=agent_results)
                stats_data["by_agent"][agent_name] = agent_tracker.get_statistics_summary()
            
            # Stats by agent and complexity
            for agent_name in agent_names:
                stats_data["by_agent_complexity"][agent_name] = {}
                for level in complexity_levels:
                    agent_level_results = [
                        r for r in self.metric_tracker.results 
                        if r.agent_name == agent_name and r.complexity_level == level
                    ]
                    if agent_level_results:
                        agent_level_tracker = MetricTracker(results=agent_level_results)
                        stats_data["by_agent_complexity"][agent_name][level] = (
                            agent_level_tracker.get_statistics_summary()
                        )
            
            with open(stats_file, "w") as f:
                json.dump(stats_data, f, indent=2, cls=NumpyEncoder)

            if self.config.verbose:
                print(f"Saved aggregated stats to: {stats_file}")

        # Save experiment config
        config_file = output_path / f"config_{timestamp}.json"
        config_data = {
            "output_dir": self.config.output_dir,
            "save_individual_results": self.config.save_individual_results,
            "save_aggregated_results": self.config.save_aggregated_results,
            "verbose": self.config.verbose,
            "complexity_levels": [
                {
                    "level": spec.level.value,
                    "num_maps": spec.num_maps,
                    "trials_per_map": spec.trials_per_map,
                    "maze_config": {
                        "space_size": spec.maze_config.space_size,
                        "grid_size": spec.maze_config.grid_size,
                        "wall_thickness": spec.maze_config.wall_thickness,
                    }
                }
                for spec in self.config.complexity_levels
            ]
        }

        with open(config_file, "w") as f:
            json.dump(config_data, f, indent=2, cls=NumpyEncoder)

        if self.config.verbose:
            print(f"Saved config to: {config_file}")
    
    def print_summary(self) -> None:
        """Print a summary of experiment results."""
        print(f"\n{'='*60}")
        print(f"Experiment Summary")
        print(f"{'='*60}")
        
        # Get unique agents and levels
        agent_names = list(set(r.agent_name for r in self.metric_tracker.results))
        
        # Overall statistics
        overall_stats = self.metric_tracker.get_statistics_summary()
        print(f"\nOverall Statistics:")
        print(f"  Total Trials: {overall_stats['total_trials']}")
        print(f"  Successful Trials: {overall_stats['successful_trials']}")
        print(f"  Success Rate: {overall_stats['success_rate']:.2%}")
        print(f"  Mean Path Length: {overall_stats['mean_path_length']:.2f}")
        print(f"  Mean Computation Time: {overall_stats['mean_computation_time']:.4f}s")
        
        # Per-agent per-complexity breakdown
        print(f"\n{'='*60}")
        print(f"Breakdown by Agent and Complexity")
        print(f"{'='*60}")
        
        for spec in self.config.complexity_levels:
            level = spec.level.value
            print(f"\n{level.upper()} (grid: {spec.maze_config.grid_size}):")
            
            for agent_name in agent_names:
                agent_level_results = [
                    r for r in self.metric_tracker.results 
                    if r.agent_name == agent_name and r.complexity_level == level
                ]
                
                if agent_level_results:
                    tracker = MetricTracker(results=agent_level_results)
                    stats = tracker.get_statistics_summary()
                    
                    print(f"  {agent_name}:")
                    print(f"    Success Rate: {stats['success_rate']:.2%} "
                          f"({stats['successful_trials']}/{stats['total_trials']})")
                    print(f"    Mean Path Length: {stats['mean_path_length']:.2f}")
                    print(f"    Mean Time: {stats['mean_computation_time']:.4f}s")

