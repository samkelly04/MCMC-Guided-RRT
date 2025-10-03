# MCMC-Guided RRT Architecture

## Overview

This project implements a Rapidly-exploring Random Tree (RRT) path planning algorithm enhanced with Markov Chain Monte Carlo (MCMC) sampling techniques. The system combines uniform random sampling with Metropolis-within-Gibbs samplers to improve path planning efficiency and robustness in uncertain environments.

## System Architecture

### Core Components

#### 1. Planning Module (`src/planners/`)
- **Tree Structure** (`tree.py`): Implements the fundamental RRT tree data structure
  - Maintains parent-child relationships for tree nodes
  - Stores node positions in 2D configuration space
  - Provides path reconstruction functionality

#### 2. Collision Detection (`src/collision.py`)
- **Point Collision**: Determines if a point lies within any obstacle
- **Segment Collision**: Checks if a line segment intersects with obstacles
- **Geometry Handling**: Uses Shapely library for robust geometric operations
  - Boundary behavior: Points on obstacle boundaries are considered free
  - Intersection detection: Any contact with obstacles (including edges/corners) is collision

#### 3. Environment Generation (`src/random_env_generator.py`)
- **Random Obstacle Placement**: Generates rectangular obstacles in configuration space
- **Non-overlapping Constraint**: Ensures obstacles don't intersect with each other
- **Configurable Parameters**: Variable number of obstacles (2-5) with random areas (4-10 units)

### Data Flow

```
1. Environment Generation
   ├── Random obstacle placement in 100x100 space
   ├── Non-overlapping validation
   └── Obstacle metadata (position, size, Shapely geometry)

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
- **Rectangular Obstacles**: Simplified geometry for computational efficiency
- **Disjoint Placement**: Prevents overlapping obstacles for valid configuration space
- **Random Generation**: Supports multiple environment instances for testing

### Integration Points

#### Testing Framework (`tests/`)
- **Collision Tests**: Comprehensive validation of collision detection edge cases
- **Boundary Conditions**: Special attention to corner cases (boundaries, multiple obstacles)
- **Geometry Validation**: Ensures consistent behavior across different input types

#### Configuration Management
- **Environment Parameters**: Configurable space dimensions and obstacle properties
- **Sampling Strategies**: Support for different random sampling approaches
- **Performance Tuning**: Adjustable parameters for RRT expansion

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
# Generate environment
obstacles = rand_env_generator(c_space)

# Initialize RRT tree
tree = Tree(start_point)

# Planning loop
while not reached_goal:
    sample = uniform_sample()  # or mcmc_sample()
    nearest = find_nearest_neighbor(sample, tree)
    if segment_is_free(nearest, sample, obstacles):
        tree.add_child(nearest_idx, sample)
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

This architecture provides a solid foundation for implementing robust path planning with enhanced sampling strategies while maintaining computational efficiency and extensibility.
