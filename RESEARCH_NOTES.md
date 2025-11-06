# Research Notes: Adaptive Goal-Localization MCMC-RRT

## Research Angle Overview

**Core Problem**: Most sampling-based planners assume perfect goal localization, but real robots operate with uncertain goal estimates from noisy sensors. This makes the comparison between goal-directed MCMC and uniform sampling unfair when both have perfect goal knowledge but only of them chooses to act on it.

**Proposed Solution**: Adaptive MCMC-guided RRT that learns a belief distribution over goal location through sequential observations during receding-horizon replanning. The MCMC target density adapts as goal belief sharpens, enabling efficient exploration even when initial goal estimates are inaccurate.

**Research Question**: Can distributionally robust adaptive goal localization improve path planning performance under goal uncertainty compared to fixed-goal and uniform sampling strategies? How can we use previous chains of correlated samples to update our beliefs about the true goal given our noisy estimates of goal localization.

---

## Receding-Horizon RRT Planning

### Core Concept
Instead of planning the entire path upfront, plan a short horizon, execute a small segment, then replan from the robot's new state. This closes the perception-planning-execution loop.

### Key Mechanics

#### Replanning Triggers
- **Time-based**: Every fixed Δt (e.g., 100-200ms)
- **Event-driven**: Immediately when:
  - Tracking error exceeds threshold (robot drifts from plan)
  - New obstacles appear near path
  - Path becomes invalid (collision detected)
  - Goal moved or time budget insufficient

#### Tree Handling Strategies
1. **Simple (Recommended First)**: Rebuild tree from scratch each cycle
   - Pros: Simple, always correct, handles large environment changes
   - Cons: Discards previous work, needs tight time budgets

2. **Warm-Started Incremental**: Re-root existing tree at current state
   - Pros: Faster convergence, smoother plan evolution
   - Cons: More complex, requires careful invalidation after map updates

#### Sampler State Management
- **Warm-start MCMC chain**: Maintain chain state across replanning cycles
- **No reseeding**: Use deterministic RNG per trial; advance naturally per cycle
- **Benefits**: Reduces "burn-in" time, maintains exploration momentum in promising regions

#### Control Execution
- **Planning horizon**: Long enough to find feasible corridor
- **Execution horizon**: Short segment (5-20% of path length or fixed time slice)
- **Validation**: Always check committed segment with footprint-aware collision before execution

#### Computational Budget
- Per-cycle planning time budget (e.g., 50-100ms) or fixed expansion count
- If no solution in budget: use last cycle's remaining path if valid, otherwise execute safe hold/evasive step

---

## Noisy Goal Estimates & Adaptive Localization

### Problem Motivation
- Real robots receive goal estimates from noisy sensors (GPS drift, vision uncertainty, landmark-based localization)
- Initial estimates may be far from true goal
- Goal location belief improves as robot gets closer and accumulates observations

### Goal Belief Representation
Maintain a probability distribution over goal location (e.g., Gaussian):

```python
P(q_goal | observations_{1:t}) = N(μ_t, Σ_t)
```

- **Initial**: Broad prior (e.g., uniform over workspace or wide Gaussian)
- **Update**: Each replan cycle receives noisy observation `goal_estimate_t ~ N(q_goal_true, Σ_obs)`
- **Fusion**: Bayesian update or weighted average to refine belief
- **Convergence**: As robot approaches and accumulates observations, belief sharpens (Σ_t decreases)

### Sequential Observation Model
- Each cycle `t`: robot receives `goal_estimate_t` with uncertainty `Σ_obs`
- Observations are independent (or can model correlation)
- True goal `q_goal_true` is fixed but unknown to planner

### Adaptive MCMC Target Density
The MCMC sampler's target distribution uses expected distance under current goal belief:

```
log π(q) = -λ_goal · E[distance(q, q_goal) | observations] 
           - λ_clearance · clearance_penalty(q)
```

Where:
- `E[distance(q, q_goal) | observations]` = expected distance under `N(μ_t, Σ_t)`
- For Gaussian belief: can compute analytically or approximate
- Hard constraints: reject out-of-bounds or colliding states (infinite cost)

---

## Cost Function to Target Density Conversion

### Basic MCMC Framework
- **Cost function** J(q): Encodes what we want (lower is better)
  - Goal proximity: `J_goal(q) = ||q - q_goal||` or `J_goal(q) = E[||q - q_goal||]` under belief
  - Clearance: `J_clearance(q) = 1 / max(clearance(q), ε)`
  - Hard constraints: `J(q) = +∞` if out-of-bounds or colliding

- **Target density**: Boltzmann transform
  ```
  π(q) ∝ exp(-J(q) / T)
  ```
  - Temperature `T` controls exploration (higher = flatter, more exploration)
  - No normalization constant needed (MCMC only uses ratios)

- **Metropolis acceptance**: For symmetric proposals (Gaussian)
  ```
  α = min(1, π(q') / π(q)) = min(1, exp(-(J(q') - J(q)) / T))
  ```

### Log-Space Stability
Use log-probabilities for numerical stability:
```
log π(q) = -J(q) / T
log α = min(0, log π(q') - log π(q))
```

### Expected Distance Under Goal Belief
For Gaussian goal belief `N(μ, Σ)`:
- **Exact**: Can compute `E[||q - q_goal||]` analytically (involves error functions)
- **Approximate**: Use distance to mean plus uncertainty penalty
  ```
  E[distance] ≈ ||q - μ|| + trace(Σ) / (2 ||q - μ||)
  ```
- **Robust**: For distributional robustness, can use worst-case or quantile distance

---

## Comparison Baselines

### 1. Uniform Sampling
- **Strategy**: Ignore goal entirely, sample uniformly
- **Expected performance**: Poor efficiency, but unbiased exploration
- **Use case**: Lower bound baseline

### 2. Fixed-Noisy MCMC
- **Strategy**: Use single noisy goal estimate (first observation), never update
- **Expected performance**: Better than uniform if estimate is good, worse if estimate is bad
- **Use case**: Baseline for adaptive methods

### 3. True-Goal MCMC (Oracle)
- **Strategy**: Use true goal location (perfect knowledge)
- **Expected performance**: Upper bound on what's possible
- **Use case**: Performance ceiling

### 4. Adaptive Goal MCMC (Our Method)
- **Strategy**: Learn goal belief through sequential observations, update target density
- **Expected performance**: Should converge faster than fixed-noisy, more robust to bad initial estimates
- **Use case**: Realistic robotics scenario

### Experimental Variables
- **Goal uncertainty**: Vary `Σ_obs` (high vs. low noise)
- **Observation frequency**: How often new goal estimates arrive
- **Initial prior quality**: Good vs. bad initial guess
- **Fusion method**: Sequential Bayesian update vs. weighted average vs. others

---

## Updated File Structure

### New Files Needed

#### Core MCMC Implementation
- `src/samplers/adaptive_goal_mcmc.py`
  - `AdaptiveGoalMCMCSampler(BaseSampler)`
  - Implements Metropolis-within-Gibbs sampling
  - Takes `GoalBelief` in state/config for target density
  - Maintains internal chain state for warm-starting

#### Goal Belief Management
- `src/rrt/goal_belief.py`
  - `GoalBelief` class
    - Attributes: `mean`, `covariance`, `observation_count`
    - Methods: `update(new_estimate, obs_covariance)`, `expected_distance(q)`
  - Bayesian update logic (sequential Kalman-style update)
  - Alternative fusion methods

#### Planner Integration
- `src/planners/rrt_planner.py`
  - `RRTPlanner` class that composes:
    - Tree (`src/rrt/tree.py`)
    - Collision checking (`src/rrt/collision.py`)
    - Steering (`src/rrt/steer.py`)
    - Nearest neighbor (`src/rrt/nearest.py`)
    - Sampler (any `BaseSampler` implementation)
  - Main planning loop: sample → nearest → steer → collision check → add node

#### Receding-Horizon Evaluation
- `src/eval/receding_horizon.py`
  - `RecedingHorizonRunner` class
  - Manages replanning loop: sense → update belief → plan → execute → repeat
  - Handles time budgets, validation, safety checks

#### Evaluation & Metrics
- `src/eval/metrics.py`
  - Metrics computation: path length, min clearance, success rate, convergence time
  - Goal belief tracking: how belief sharpens over time
  - Path quality metrics under uncertainty

- `src/eval/runner.py`
  - Experiment runner for batch evaluation
  - Logs to CSV (tidy format, one row per rollout)
  - Handles seed management for reproducibility

#### Evaluation Scripts
- `experiments/adaptive_goal_experiment.py`
  - Main experiment script
  - Configures noise levels, observation frequencies, fusion methods
  - Runs comparisons across all baselines

### Modified Files

#### Minimal Interface Extension
- `src/samplers/base.py`
  - Extend `sample()` signature:
    ```python
    def sample(self, bounds: np.ndarray, state: dict | None = None) -> np.ndarray:
    ```
  - `state` dict can contain: `goal_belief`, `env`, `current_tree_state`, etc.
  - Uniform sampler ignores `state` (backward compatible)
  - MCMC sampler uses `state` for adaptive target density

### Existing Files (No Changes Needed)
- `src/rrt/tree.py` - Tree data structure
- `src/rrt/collision.py` - Collision checking
- `src/rrt/steer.py` - Steering logic
- `src/rrt/nearest.py` - Nearest neighbor search
- `src/rrt/goal.py` - Goal utilities (can coexist with GoalBelief)
- `src/env/generator.py` - Maze/environment generation
- `src/samplers/uniform.py` - Uniform baseline sampler

### Testing Structure
- `tests/test_goal_belief.py` - Goal belief update logic, fusion methods
- `tests/test_adaptive_mcmc.py` - MCMC sampler behavior, chain state, acceptance ratios
- `tests/test_receding_horizon.py` - Replanning loop, belief convergence
- `tests/smoke_adaptive_goal.py` - 200-trial smoke test with golden seeds

---

## Implementation Order

1. **Goal Belief** (`src/rrt/goal_belief.py`)
   - Simple Gaussian belief with Bayesian update
   - Expected distance computation
   - Unit tests for update logic

2. **Base Sampler Interface** (`src/samplers/base.py`)
   - Add optional `state` parameter
   - Update uniform sampler to accept (ignore) it

3. **Adaptive MCMC Sampler** (`src/samplers/adaptive_goal_mcmc.py`)
   - Metropolis-within-Gibbs implementation
   - Target density using goal belief
   - Warm-start chain state management
   - Unit tests for bounds, acceptance, reproducibility

4. **RRT Planner** (`src/planners/rrt_planner.py`)
   - Composes existing RRT components
   - Takes any sampler (uniform or MCMC)
   - Integration tests

5. **Receding-Horizon Runner** (`src/eval/receding_horizon.py`)
   - Replanning loop
   - Goal observation injection
   - Belief tracking
   - Safety validation

6. **Evaluation Framework** (`src/eval/metrics.py`, `src/eval/runner.py`)
   - Metrics computation
   - CSV logging
   - Batch experiment runner

7. **Experiments** (`experiments/adaptive_goal_experiment.py`)
   - Compare all baselines
   - Vary noise levels and observation frequencies
   - Generate results and visualizations

---

## Key Research Questions

1. **Convergence Analysis**: How quickly does goal belief converge? How does planning performance improve as belief sharpens?

2. **Robustness**: How does adaptive MCMC handle bad initial estimates? Can it recover from outliers?

3. **Comparison**: Under what conditions does adaptive MCMC outperform fixed-noisy and uniform sampling?

4. **Fusion Methods**: Which goal belief update strategy works best? Sequential Bayesian vs. weighted average vs. particle filter?

5. **Distributional Robustness**: Does using expected distance under uncertainty improve success rates compared to using single noisy estimates?

6. **Computational Tradeoffs**: What's the overhead of belief updates and expected distance computation? Is it worth it?

---

## Success Metrics

- **Planning Efficiency**: Success rate, path length, number of expansions
- **Robustness**: Performance degradation under goal uncertainty
- **Convergence**: Speed at which goal belief sharpens and planning improves
- **Recovery**: Ability to recover from bad initial goal estimates
- **Comparison**: Relative performance vs. baselines across noise levels

---

## Notes on Distributional Robustness

The key insight is that we're not just biasing toward a single goal estimate, but biasing toward a distribution over goal locations. This makes the planner naturally robust to uncertainty:

- **Expected distance**: Favors regions that are good on average across possible goal locations
- **Variance consideration**: Can penalize high-variance regions (prefer conservative paths)
- **Adaptive refinement**: As belief sharpens, planner becomes more focused

This is fundamentally different from using a single noisy estimate, which can be misled by bad observations.

