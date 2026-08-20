"""Loaders for experimental trial results.

This module provides functions to load JSON trial results and reconstruct
numpy arrays for analysis.
"""

import json
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Dict, Any
from datetime import datetime

import numpy as np


@dataclass
class LoadedTrialResult:
    """Loaded trial result with numpy arrays reconstructed.

    This dataclass mirrors TrialResult but with all arrays reconstructed
    from JSON serialization.
    """

    agent_name: str
    complexity_level: str
    map_seed: Optional[int]
    success: bool
    steps_taken: int
    nodes_expanded: int
    path_length: float
    computation_time: float
    timestamp: datetime

    # Adaptive navigation metrics
    num_replans: Optional[int] = None
    final_belief_mean: Optional[np.ndarray] = None
    final_belief_covariance: Optional[np.ndarray] = None
    final_belief_entropy: Optional[float] = None
    belief_convergence: Optional[float] = None

    # Additional metrics and histories
    belief_history: Optional[List[Dict[str, Any]]] = None
    entropy_history: Optional[List[float]] = None
    trial_seed: Optional[int] = None
    num_steps: Optional[int] = None
    distance_to_goal: Optional[float] = None


def load_trials(json_path: str) -> List[LoadedTrialResult]:
    """Load trial results from JSON file and reconstruct numpy arrays.

    Args:
        json_path: Path to JSON file containing trial results.

    Returns:
        List of LoadedTrialResult objects with numpy arrays reconstructed.

    Raises:
        FileNotFoundError: If json_path doesn't exist.
        json.JSONDecodeError: If file contains invalid JSON.
    """
    path = Path(json_path)
    if not path.exists():
        raise FileNotFoundError(f"Trial results file not found: {json_path}")

    with open(path, 'r') as f:
        data = json.load(f)

    results = []
    for trial_dict in data:
        # Extract additional metrics
        additional = trial_dict.get('additional_metrics', {})

        # Reconstruct numpy arrays
        final_belief_mean = None
        if trial_dict.get('final_belief_mean') is not None:
            final_belief_mean = np.array(trial_dict['final_belief_mean'])

        final_belief_covariance = None
        if trial_dict.get('final_belief_covariance') is not None:
            final_belief_covariance = np.array(trial_dict['final_belief_covariance'])

        # Reconstruct belief history with numpy arrays
        belief_history = None
        if 'belief_history' in additional:
            belief_history = []
            for belief_dict in additional['belief_history']:
                reconstructed = {
                    'mean': np.array(belief_dict['mean']),
                    'covariance': np.array(belief_dict['covariance']),
                    'entropy': belief_dict['entropy'],
                    'observation_count': belief_dict['observation_count'],
                }
                belief_history.append(reconstructed)

        # Parse timestamp
        timestamp = datetime.fromisoformat(trial_dict['timestamp'])

        result = LoadedTrialResult(
            agent_name=trial_dict['agent_name'],
            complexity_level=trial_dict['complexity_level'],
            map_seed=trial_dict.get('map_seed'),
            success=trial_dict['success'],
            steps_taken=trial_dict['steps_taken'],
            nodes_expanded=trial_dict['nodes_expanded'],
            path_length=trial_dict['path_length'],
            computation_time=trial_dict['computation_time'],
            timestamp=timestamp,
            num_replans=trial_dict.get('num_replans'),
            final_belief_mean=final_belief_mean,
            final_belief_covariance=final_belief_covariance,
            final_belief_entropy=trial_dict.get('final_belief_entropy'),
            belief_convergence=trial_dict.get('belief_convergence'),
            belief_history=belief_history,
            entropy_history=additional.get('entropy_history'),
            trial_seed=additional.get('trial_seed'),
            num_steps=additional.get('num_steps'),
            distance_to_goal=additional.get('distance_to_goal'),
        )
        results.append(result)

    return results


def filter_by_agent(results: List[LoadedTrialResult], agent_name: str) -> List[LoadedTrialResult]:
    """Filter results to only those from a specific agent.

    Args:
        results: List of trial results.
        agent_name: Name of agent to filter for.

    Returns:
        Filtered list containing only trials from the specified agent.
    """
    return [r for r in results if r.agent_name == agent_name]


def filter_by_complexity(results: List[LoadedTrialResult], level: str) -> List[LoadedTrialResult]:
    """Filter results to only those from a specific complexity level.

    Args:
        results: List of trial results.
        level: Complexity level to filter for (e.g., 'easy', 'medium', 'hard').

    Returns:
        Filtered list containing only trials from the specified complexity level.
    """
    return [r for r in results if r.complexity_level == level]


def filter_successful(results: List[LoadedTrialResult]) -> List[LoadedTrialResult]:
    """Filter results to only successful trials.

    Args:
        results: List of trial results.

    Returns:
        Filtered list containing only successful trials.
    """
    return [r for r in results if r.success]


def filter_adaptive(results: List[LoadedTrialResult]) -> List[LoadedTrialResult]:
    """Filter results to only those with adaptive telemetry.

    Args:
        results: List of trial results.

    Returns:
        Filtered list containing only trials with num_replans data.
    """
    return [r for r in results if r.num_replans is not None]


def get_unique_agents(results: List[LoadedTrialResult]) -> List[str]:
    """Get list of unique agent names in results.

    Args:
        results: List of trial results.

    Returns:
        Sorted list of unique agent names.
    """
    return sorted(list(set(r.agent_name for r in results)))


def get_unique_complexity_levels(results: List[LoadedTrialResult]) -> List[str]:
    """Get list of unique complexity levels in results.

    Args:
        results: List of trial results.

    Returns:
        List of unique complexity levels in order: easy, medium, hard.
    """
    levels = list(set(r.complexity_level for r in results))
    # Sort by standard order
    order = {'easy': 0, 'medium': 1, 'hard': 2}
    return sorted(levels, key=lambda x: order.get(x, 99))
