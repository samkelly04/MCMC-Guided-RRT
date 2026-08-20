"""Configuration dataclasses for evaluation experiments."""

from dataclasses import dataclass, field
from enum import Enum
from typing import Tuple, Optional


class ComplexityLevel(str, Enum):
    """Maze complexity levels for systematic evaluation.
    
    These levels define different tiers of maze difficulty, allowing
    experiments to test agent performance across a range of challenges.
    """
    
    EASY = "easy"
    MEDIUM = "medium"
    HARD = "hard"
    EXTREME = "extreme"


@dataclass
class MazeGenerationConfig:
    """Configuration for generating individual mazes.
    
    This config defines the parameters used to generate a single maze instance,
    including dimensions, wall density, and random seed for reproducibility.
    """
    
    space_size: Tuple[float, float] = (100.0, 100.0)
    """Workspace dimensions (width, height) in continuous coordinates."""
    
    grid_size: Tuple[int, int] = (10, 10)
    """Grid dimensions (rows, cols) for maze generation."""
    
    wall_thickness: float = 1.0
    """Thickness of wall segments."""
    
    seed: Optional[int] = None
    """Random seed for maze generation. None for non-deterministic."""
    
    def __post_init__(self):
        """Validate configuration parameters."""
        if self.space_size[0] <= 0 or self.space_size[1] <= 0:
            raise ValueError("space_size must have positive dimensions")
        if self.grid_size[0] <= 0 or self.grid_size[1] <= 0:
            raise ValueError("grid_size must have positive dimensions")
        if self.wall_thickness <= 0:
            raise ValueError("wall_thickness must be positive")


@dataclass
class ComplexitySpec:
    """Specification for a complexity level.
    
    Defines the maze generation parameters and testing scale for a specific
    complexity tier (e.g., Easy, Medium, Hard).
    """
    
    level: ComplexityLevel
    """Complexity level identifier."""
    
    maze_config: MazeGenerationConfig
    """Maze generation parameters for this complexity level."""
    
    num_maps: int = 50
    """Number of unique map seeds/layouts to generate per complexity level."""
    
    trials_per_map: int = 1
    """Number of times to run the agent on each specific map.
    
    Useful for stochastic agents where multiple runs on the same map
    may produce different results.
    """
    
    def __post_init__(self):
        """Validate complexity specification."""
        if self.num_maps <= 0:
            raise ValueError("num_maps must be positive")
        if self.trials_per_map <= 0:
            raise ValueError("trials_per_map must be positive")


@dataclass
class ExperimentConfig:
    """Complete configuration for an evaluation experiment.
    
    This is the main configuration object that defines all parameters for
    running a comprehensive evaluation of maze-solving agents across multiple
    complexity levels.
    """
    
    complexity_levels: list[ComplexitySpec] = field(default_factory=list)
    """List of complexity specifications to test."""
    
    output_dir: str = "results"
    """Directory path for saving experiment results."""
    
    save_individual_results: bool = True
    """Whether to save results for each individual trial."""
    
    save_aggregated_results: bool = True
    """Whether to save aggregated statistics across all trials."""
    
    verbose: bool = True
    """Whether to print progress information during experiments."""
    
    def __post_init__(self):
        """Validate experiment configuration."""
        if not self.complexity_levels:
            raise ValueError("At least one complexity level must be specified")
        if not self.output_dir:
            raise ValueError("output_dir cannot be empty")
    
    def get_total_trials(self) -> int:
        """Calculate total number of trials across all complexity levels.
        
        Returns:
            Total number of trials = sum(num_maps * trials_per_map) for each level.
        """
        return sum(spec.num_maps * spec.trials_per_map for spec in self.complexity_levels)
    
    def get_total_maps(self) -> int:
        """Calculate total number of unique maps across all complexity levels.
        
        Returns:
            Total number of unique maps = sum(num_maps) for each level.
        """
        return sum(spec.num_maps for spec in self.complexity_levels)
