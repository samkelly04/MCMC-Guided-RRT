# MCMC-Guided RRT Architecture

## Overview

This project implements a Rapidly-exploring Random Tree (RRT) path planning algorithm enhanced with Markov Chain Monte Carlo (MCMC) sampling techniques. The system combines uniform random sampling with Metropolis-within-Gibbs samplers to improve path planning efficiency and robustness in uncertain environments.

## System Architecture

### Core Components

#### 1. RRT Module (`src/rrt/`)
- **Tree Structure** (`tree.py`): Implements the fundamental RRT tree data structure
  - Maintains parent-child relationships for tree nodes
  - Stores node positions in 2D configuration space
  - Provides path reconstruction functionality
- **Collision Detection** (`collision.py`): Geometric collision checking utilities
  - **Point Collision**: Determines if a point lies within any obstacle
  - **Segment Collision**: Checks if a line segment intersects with obstacles
  - **Geometry Handling**: Uses Shapely library for robust geometric operations
    - Boundary behavior: Points on obstacle boundaries are considered free
    - Intersection detection: Any contact with obstacles (including edges/corners) is collision

#### 2. Environment Generation (`src/env/generator.py`)
- **Maze-Based Generation**: Generates maze-like environments using grid-based random walk algorithm
- **Grid-Based Design**: Uses a grid to systematically place walls that form corridors and passages
  - Grid cells represent potential wall locations, not sampling constraints
  - Grid is a design tool only - discarded after wall placement
- **Connectivity Guarantee**: Random walk algorithm ensures at least one path exists from start to goal
  - Algorithm starts from initial cell and randomly walks to neighbors
  - Removes walls between connected cells during walk
  - Ensures all cells in path are reachable from start
- **Configurable Parameters**:
  - `space_size`: Total workspace dimensions (width, height) - e.g., (100, 100)
  - `grid_size`: Number of cells in grid (rows, cols) - e.g., (20, 20)
  - `wall_thickness`: Width of wall segments - e.g., 1.0
  - `seed`: Random seed for reproducibility
- **Continuous State Space**: Grid is only used for generation; RRT samples from continuous space
  - Walls are converted to Shapely rectangles in continuous coordinates
  - RRT can sample any point (e.g., (1.5, 7.0)) not just grid points
  - Collision checking operates in continuous space via Shapely

### Data Flow

```
1. Environment Generation
   ├── Grid initialization (configurable dimensions)
   ├── Random walk algorithm to create maze structure
   ├── Wall placement based on grid connectivity
   ├── Conversion to Shapely rectangles (continuous space)
   └── Obstacle list ready for RRT planning

2. RRT Planning Loop
   ├── Tree initialization with start point
   ├── Random sampling (uniform or MCMC-guided)
   ├── Nearest neighbor search
   ├── Collision-free edge generation
   └── Tree expansion until goal reached

3. Path Extraction
   ├── Backtrack from goal to root
   └── Path validation in noisy environment
```

### Key Design Decisions

#### Collision Detection Strategy
- **Boundary Treatment**: Uses Shapely's `contains()` method where boundaries are excluded
- **Segment Validation**: Uses `intersects()` for line segments to catch any contact
- **Performance**: Optimized for frequent collision checks during RRT expansion

#### Tree Representation
- **Efficient Storage**: NumPy arrays for positions and parent indices
- **2D Configuration**: Focused on planar path planning
- **Dynamic Growth**: Supports incremental tree expansion

#### Environment Modeling
- **Maze-Based Structure**: Grid-based random walk creates structured maze-like environments
- **Wall Representation**: Walls are rectangular Shapely geometries in continuous space
- **Connectivity Guarantee**: Random walk algorithm ensures solvable mazes (at least one path exists)
- **Configurable Complexity**: Grid size and space size are configurable for testing
  - Different grid sizes test algorithm performance across complexity levels
  - Same dimensions with different seeds test robustness on different layouts
- **Continuous Sampling**: Grid is generation tool only; RRT operates in continuous space
  - Grid cells help place walls systematically
  - Once walls are Shapely rectangles, grid is irrelevant
  - RRT can sample any continuous point, not limited to grid cells

### Integration Points

#### Testing Framework (`tests/`)
- **Collision Tests**: Comprehensive validation of collision detection edge cases
- **Boundary Conditions**: Special attention to corner cases (boundaries, multiple obstacles)
- **Geometry Validation**: Ensures consistent behavior across different input types

#### Configuration Management
- **Environment Parameters**: 
  - Configurable space dimensions (`space_size`) and grid complexity (`grid_size`)
  - Wall thickness and random seed for reproducibility
  - Enables systematic testing across different maze complexities
- **Sampling Strategies**: Support for different random sampling approaches
- **Performance Tuning**: Adjustable parameters for RRT expansion
- **Testing Strategy**:
  - Dimension variation: Test different grid sizes to assess complexity scaling
  - Randomness variation: Test different seeds on same dimensions for robustness
  - Reproducibility: Fixed seeds enable controlled experiments and comparisons

#### 3. Evaluation Framework (`src/eval/`, `src/core/`, `src/config/`)

The evaluation framework provides a systematic, scalable architecture for testing maze-solving agents across multiple complexity levels and configurations.

##### Core Interfaces (`src/core/`)

- **MazeEnvironment** (`environment.py`): Abstract base class defining the interface for maze environments
  - `get_start()`: Returns starting position
  - `get_goal()`: Returns goal position
  - `is_wall(position)`: Checks if a position contains a wall/obstacle
  - `get_neighbors(position)`: Returns valid neighboring positions
  - `get_bounds()`: Returns workspace bounds
  - `get_dimensions()`: Returns maze dimensions
  - `is_valid_position(position)`: Validates position (in bounds and free)

- **MazeAgent** (`agent.py`): Abstract base class defining the interface for maze-solving agents
  - `solve(environment)`: Main entry point - solves maze and returns path
  - `get_next_move(current_position, environment)`: Step-by-step navigation
  - `reset(seed)`: Resets agent state (important for stochastic agents)
  - `get_name()`: Returns human-readable agent name

##### Configuration System (`src/config/`)

- **ComplexityLevel** (`experiment_config.py`): Enum defining complexity tiers
  - `EASY`: Sparse walls, simple layouts
  - `MEDIUM`: Moderate complexity
  - `HARD`: Dense walls, complex layouts
  - `EXTREME`: Maximum difficulty

- **MazeGenerationConfig**: Dataclass for maze generation parameters
  - `space_size`: Workspace dimensions (width, height)
  - `grid_size`: Grid dimensions (rows, cols)
  - `wall_thickness`: Thickness of wall segments
  - `seed`: Random seed for reproducibility

- **ComplexitySpec**: Specification for a complexity level
  - `level`: ComplexityLevel identifier
  - `maze_config`: MazeGenerationConfig for this level
  - `num_maps`: Number of unique map seeds/layouts per complexity level
  - `trials_per_map`: Number of runs per map (for stochastic agents)

- **ExperimentConfig**: Complete experiment configuration
  - `complexity_levels`: List of ComplexitySpec objects
  - `output_dir`: Directory for saving results
  - `save_individual_results`: Whether to save per-trial results
  - `save_aggregated_results`: Whether to save aggregated statistics
  - `verbose`: Whether to print progress information
  - Methods: `get_total_trials()`, `get_total_maps()`

##### Metric Tracking (`src/eval/metrics.py`)

- **TrialResult**: Dataclass capturing metrics from a single trial
  - `agent_name`: Name of evaluated agent
  - `complexity_level`: Complexity level string
  - `map_seed`: Random seed used for maze generation
  - `success`: Boolean indicating if path was found
  - `steps_taken`: Number of steps/moves
  - `nodes_expanded`: Nodes expanded during search
  - `path_length`: Total path length (Euclidean distance)
  - `computation_time`: Time taken in seconds
  - `path`: Actual path found (list of positions)
  - `timestamp`: When trial was completed
  - `additional_metrics`: Dictionary for custom metrics

- **MetricTracker**: Class for tracking and aggregating metrics
  - `record_result(result)`: Records a new trial result
  - `get_success_rate(complexity_level)`: Calculates success rate
  - `get_mean_steps(complexity_level)`: Mean steps for successful trials
  - `get_mean_computation_time(complexity_level)`: Mean computation time
  - `get_mean_path_length(complexity_level)`: Mean path length
  - `get_statistics_summary(complexity_level)`: Comprehensive statistics
  - `clear()`: Clears all recorded results

##### Evaluator (`src/eval/evaluator.py`)

- **Evaluator**: Main class orchestrating experiments
  - `__init__(config)`: Initializes with ExperimentConfig
  - `setup_experiment()`: Prepares experiment environment
  - `generate_mazes(complexity_spec)`: Generates mazes for a complexity level
  - `run_trial(agent, environment, complexity_level, map_seed)`: Runs single trial
  - `record_result(result)`: Records trial result in metric tracker
  - `run_experiment(agents)`: Main entry point - runs complete experiment
  - `save_results(filepath)`: Saves results to disk
  - `print_summary()`: Prints aggregated statistics

##### Evaluation Framework Architecture

```
Experiment Flow:
1. Configuration Setup
   ├── Define complexity levels (Easy, Medium, Hard)
   ├── Set num_maps per level (e.g., 50 maps)
   ├── Set trials_per_map (e.g., 1 for deterministic, 10 for stochastic)
   └── Configure output directory and logging

2. Maze Generation
   ├── For each complexity level:
   │   ├── Generate num_maps unique mazes (different seeds)
   │   └── Each maze uses complexity_spec.maze_config parameters
   └── All mazes implement MazeEnvironment interface

3. Agent Evaluation
   ├── For each agent:
   │   ├── For each complexity level:
   │   │   ├── For each maze:
   │   │   │   ├── For each trial (trials_per_map):
   │   │   │   │   ├── Reset agent state
   │   │   │   │   ├── Run agent.solve() or step-by-step execution
   │   │   │   │   ├── Collect metrics (steps, time, success, path)
   │   │   │   │   └── Record TrialResult
   │   │   │   └── Aggregate results for this maze
   │   │   └── Aggregate results for this complexity level
   │   └── Aggregate results for this agent
   └── Generate overall statistics

4. Result Analysis
   ├── Calculate success rates per complexity level
   ├── Compute mean steps, computation time, path length
   ├── Generate per-agent and per-complexity summaries
   └── Save results (individual trials + aggregated stats)
```

##### Key Design Principles

- **Modularity**: Clear separation between interfaces (core), configuration (config), and execution (eval)
- **Type Safety**: Extensive use of Python 3.10+ type hints and dataclasses
- **Extensibility**: Abstract base classes allow easy addition of new agents/environments
- **Reproducibility**: Seed-based maze generation and agent reset support
- **Scalability**: Designed to handle large-scale experiments (hundreds of maps, multiple agents)
- **Flexibility**: Supports both deterministic and stochastic agents via trials_per_map

##### Integration with Existing Code

The evaluation framework is designed to work with existing components:
- **Environment Generation**: `src/env/generator.py` can be wrapped to implement `MazeEnvironment`
- **RRT Planner**: `src/rrt/planner.py` can be wrapped to implement `MazeAgent`
- **Samplers**: `src/samplers/` can be used within agent implementations

### Future Extensions

#### MCMC Integration
- **Metropolis-within-Gibbs**: Enhanced sampling for better exploration
- **Adaptive Sampling**: Dynamic adjustment based on environment complexity
- **Multi-modal Sampling**: Support for different sampling distributions

#### Robustness Features
- **Noisy Environment**: Gaussian noise simulation for obstacle uncertainty, specifically on obstacle size and position
- **Performance Evaluation**: 100-rollout validation in noisy conditions
- **Planning vs. Execution**: Distinction between planning world and noisy world

### Dependencies

#### Core Libraries
- **NumPy**: Numerical operations and array management
- **Shapely**: Geometric computations and collision detection
- **Random**: Environment generation and sampling

#### Development Tools
- **Pytest**: Testing framework
- **Ruff**: Code formatting and linting
- **UV**: Python package management

### Performance Considerations

#### Computational Complexity
- **Collision Checks**: O(n) per check where n is number of obstacles
- **Tree Operations**: O(log n) for nearest neighbor search (with spatial indexing)
- **Memory Usage**: Linear growth with tree size

#### Optimization Opportunities
- **Spatial Indexing**: For faster nearest neighbor queries
- **Lazy Collision Checking**: Defer expensive operations
- **Parallel Sampling**: Concurrent MCMC chains for exploration

### Error Handling

#### Robustness Measures
- **Input Validation**: Type checking and dimension validation
- **Boundary Cases**: Proper handling of edge conditions
- **Graceful Degradation**: Fallback strategies for sampling failures

#### Testing Coverage
- **Unit Tests**: Individual component validation
- **Integration Tests**: End-to-end planning pipeline
- **Property Tests**: Randomized testing for edge cases

## Usage Patterns

### Basic Planning
```python
# Generate maze environment
obstacles = generate_maze(
    space_size=(100, 100),
    grid_size=(20, 20),
    wall_thickness=1.0,
    seed=42
)

# Initialize RRT tree
tree = Tree(start_point)

# Planning loop
while not reached_goal:
    sample = sampler.sample(bounds)  # Continuous sampling (e.g., (47.3, 23.7))
    nearest_idx = find_nearest(tree, sample)
    q_new = steer(q_near, sample, step_size, bounds)
    if segment_is_free(q_near, q_new, obstacles):  # Continuous collision check
        tree.add_node(nearest_idx, q_new)
```

### Environment Validation
```python
# Check point validity
if point_is_free(point, obstacles):
    # Safe to use point

# Validate path segment
if segment_is_free(point_a, point_b, obstacles):
    # Safe edge for tree expansion
```
