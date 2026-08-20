"""Metric tracking for maze-solving agent evaluation."""

from dataclasses import dataclass, field
from typing import List, Optional
from datetime import datetime
import numpy as np


@dataclass
class TrialResult:
    """Results from a single trial of an agent solving a maze.
    
    This dataclass captures all relevant metrics for one run of an agent
    on a specific maze instance.
    """
    
    agent_name: str
    """Name of the agent that was evaluated."""
    
    complexity_level: str
    """Complexity level of the maze (e.g., 'easy', 'medium', 'hard')."""
    
    map_seed: Optional[int]
    """Random seed used to generate the maze."""
    
    success: bool
    """Whether the agent successfully found a path to the goal."""
    
    steps_taken: int
    """Number of steps/moves taken by the agent."""
    
    nodes_expanded: int = 0
    """Number of nodes expanded during search (for tree/graph-based algorithms)."""
    
    path_length: float = 0.0
    """Total length of the path found (Euclidean distance sum)."""
    
    computation_time: float = 0.0
    """Time taken to solve the maze in seconds."""
    
    path: Optional[List[np.ndarray]] = None
    """The actual path found (list of positions). None if no path found."""

    # Adaptive navigation metrics (populated only for adaptive agents)
    num_replans: Optional[int] = None
    """Number of replanning events triggered during adaptive navigation."""

    final_belief_mean: Optional[np.ndarray] = None
    """Final belief mean position (goal estimate after navigation)."""

    final_belief_covariance: Optional[np.ndarray] = None
    """Final belief covariance matrix (uncertainty after navigation)."""

    final_belief_entropy: Optional[float] = None
    """Final belief entropy (trace of covariance matrix)."""

    belief_convergence: Optional[float] = None
    """Distance from final belief mean to true goal (accuracy metric)."""

    timestamp: datetime = field(default_factory=datetime.now)
    """Timestamp when this trial was completed."""

    additional_metrics: dict = field(default_factory=dict)
    """Dictionary for storing agent-specific or custom metrics."""
    
    def __post_init__(self):
        """Validate trial result data."""
        if self.steps_taken < 0:
            raise ValueError("steps_taken cannot be negative")
        if self.nodes_expanded < 0:
            raise ValueError("nodes_expanded cannot be negative")
        if self.path_length < 0:
            raise ValueError("path_length cannot be negative")
        if self.computation_time < 0:
            raise ValueError("computation_time cannot be negative")


@dataclass
class MetricTracker:
    """Tracks and aggregates metrics across multiple trials.
    
    This class collects trial results and provides methods to compute
    aggregate statistics (mean, std, success rate, etc.) across trials.
    """
    
    results: List[TrialResult] = field(default_factory=list)
    """List of all trial results collected so far."""
    
    def record_result(self, result: TrialResult) -> None:
        """Record a new trial result.
        
        Args:
            result: TrialResult to add to the collection.
        """
        self.results.append(result)
    
    def get_success_rate(self, complexity_level: Optional[str] = None) -> float:
        """Calculate success rate across trials.
        
        Args:
            complexity_level: Optional filter by complexity level.
                           If None, calculates across all trials.
        
        Returns:
            Success rate as a float between 0.0 and 1.0.
        """
        filtered = self._filter_by_complexity(complexity_level)
        if not filtered:
            return 0.0
        successful = sum(1 for r in filtered if r.success)
        return successful / len(filtered)
    
    def get_mean_steps(self, complexity_level: Optional[str] = None) -> float:
        """Calculate mean number of steps across successful trials.
        
        Args:
            complexity_level: Optional filter by complexity level.
        
        Returns:
            Mean steps taken, or 0.0 if no successful trials.
        """
        filtered = self._filter_successful(complexity_level)
        if not filtered:
            return 0.0
        return sum(r.steps_taken for r in filtered) / len(filtered)
    
    def get_mean_computation_time(self, complexity_level: Optional[str] = None) -> float:
        """Calculate mean computation time across trials.
        
        Args:
            complexity_level: Optional filter by complexity level.
        
        Returns:
            Mean computation time in seconds.
        """
        filtered = self._filter_by_complexity(complexity_level)
        if not filtered:
            return 0.0
        return sum(r.computation_time for r in filtered) / len(filtered)
    
    def get_mean_path_length(self, complexity_level: Optional[str] = None) -> float:
        """Calculate mean path length across successful trials.
        
        Args:
            complexity_level: Optional filter by complexity level.
        
        Returns:
            Mean path length, or 0.0 if no successful trials.
        """
        filtered = self._filter_successful(complexity_level)
        if not filtered:
            return 0.0
        return sum(r.path_length for r in filtered) / len(filtered)
    
    def get_statistics_summary(self, complexity_level: Optional[str] = None) -> dict:
        """Get comprehensive statistics summary.
        
        Args:
            complexity_level: Optional filter by complexity level.
        
        Returns:
            Dictionary containing aggregated statistics:
            - success_rate: float
            - mean_steps: float
            - mean_computation_time: float
            - mean_path_length: float
            - total_trials: int
            - successful_trials: int
        """
        filtered = self._filter_by_complexity(complexity_level)
        successful = self._filter_successful(complexity_level)
        
        return {
            "success_rate": self.get_success_rate(complexity_level),
            "mean_steps": self.get_mean_steps(complexity_level),
            "mean_computation_time": self.get_mean_computation_time(complexity_level),
            "mean_path_length": self.get_mean_path_length(complexity_level),
            "total_trials": len(filtered),
            "successful_trials": len(successful),
        }
    
    def _filter_by_complexity(self, complexity_level: Optional[str]) -> List[TrialResult]:
        """Filter results by complexity level."""
        if complexity_level is None:
            return self.results
        return [r for r in self.results if r.complexity_level == complexity_level]
    
    def _filter_successful(self, complexity_level: Optional[str] = None) -> List[TrialResult]:
        """Filter to only successful trials."""
        filtered = self._filter_by_complexity(complexity_level)
        return [r for r in filtered if r.success]
    
    def clear(self) -> None:
        """Clear all recorded results."""
        self.results.clear()
