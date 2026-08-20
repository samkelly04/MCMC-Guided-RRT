# FINAL_SPEC.md
## Absolute Source of Truth for MCMC-Guided-RRT Implementation

This document describes **only what is implemented** in the Python codebase. All references cite specific variable names and line numbers.

---

## 1. Sensing: Goal Observation Model

**File**: `src/rrt/goal_sensor.py`

### Mathematical Model (`observe` method, lines 46-71)

The sensor returns a noisy observation of the true goal location:

```python
observation = goal + self._rng.normal(0.0, sigma, size=goal.shape)  # Line 69
covariance = np.eye(goal.shape[0]) * (sigma**2)                     # Line 70
```

Where `sigma` is computed via distance-dependent noise scaling.

### Noise Scaling (`_noise_std` method, lines 73-79)

The standard deviation decreases as the robot approaches the goal:

```python
distance = np.linalg.norm(goal - robot_pos)                          # Line 66 (in observe)
alpha = min(distance / max(config.distance_scale, 1e-6), 1.0)       # Line 77
sigma = config.near_std + (config.far_std - config.near_std) * alpha # Line 79
```

**Behavior**:
- When `distance >= distance_scale`: `alpha = 1.0` → `sigma = far_std` (default: 8.0)
- When `distance = 0`: `alpha = 0.0` → `sigma = near_std` (default: 0.5)
- Linear interpolation between these extremes

**Default Configuration** (lines 23-26):
- `far_std = 8.0`
- `near_std = 0.5`
- `distance_scale = 50.0`

---

## 2. Belief Update: Kalman Fusion

**File**: `src/rrt/goal_belief.py`

### Update Mechanism (`update` method, lines 46-71)

The belief is updated using standard Kalman filter equations:

```python
innovation = obs - self.mean                                    # Line 64
innovation_cov = self.covariance + obs_cov                      # Line 65
gain = self.covariance @ np.linalg.inv(innovation_cov)          # Line 66

self.mean = self.mean + gain @ innovation                       # Line 68
self.covariance = (identity - gain) @ self.covariance           # Line 70
self.observation_count += 1                                     # Line 71
```

**Variables Modified**:
- `self.mean`: Updated belief mean (posterior estimate of goal location)
- `self.covariance`: Updated belief covariance (uncertainty)
- `self.observation_count`: Incremented counter

**Implementation Detail**: This is the standard Joseph form Kalman update. The covariance update uses `(I - K)Σ` rather than the symmetrized Joseph form `(I - K)Σ(I - K)^T + KRK^T`.

---

## 3. Target Selection: Robust Sampling Strategy

**File**: `src/rrt/navigation.py`

### Planning Entry Point (lines 179-187)

The navigation loop does **NOT** plan directly to `belief.mean`. Instead:

```python
# Get valid target from belief distribution
target = get_valid_target(current_belief, obstacles, bounds, num_samples=100)  # Line 182

# Plan using valid target
planner = RRTPlanner(
    start=current_pose,
    goal=target,  # Plan toward valid target (not belief.mean directly)  # Line 187
    ...
)
```

### Valid Target Selection (`get_valid_target` function, lines 21-79)

Multi-stage fallback strategy:

**Stage 1**: Check if clamped belief mean is valid (lines 44-47)
```python
clamped_mean = np.clip(belief.mean, bounds[0], bounds[1])  # Line 45
if point_is_free(clamped_mean, obstacles):
    return clamped_mean  # Line 47
```

**Stage 2**: Sample from belief distribution and filter valid points (lines 49-66)
```python
samples = belief.sample(num_samples)  # Line 50

valid_samples = []
for sample in samples:
    if np.all(bounds[0] <= sample) and np.all(sample <= bounds[1]):  # Line 56
        if point_is_free(sample, obstacles):                          # Line 58
            valid_samples.append(sample)                              # Line 59

if len(valid_samples) > 0:
    valid_samples_array = np.array(valid_samples)                                     # Line 63
    distances_to_mean = np.linalg.norm(valid_samples_array - belief.mean, axis=1)    # Line 64
    closest_idx = np.argmin(distances_to_mean)                                        # Line 65
    return valid_samples_array[closest_idx]                                           # Line 66
```

**Stage 3**: Local grid search near clamped mean (lines 68-75)
```python
search_radius = min(np.ptp(bounds, axis=0)) * 0.1  # 10% of smallest dimension  # Line 70
for _ in range(50):  # Try 50 random points
    offset = np.random.uniform(-search_radius, search_radius, size=belief.dimension)  # Line 72
    candidate = np.clip(clamped_mean + offset, bounds[0], bounds[1])                  # Line 73
    if point_is_free(candidate, obstacles):
        return candidate  # Line 75
```

**Stage 4**: Absolute fallback (lines 77-79)
```python
return clamped_mean  # Even if in collision - will trigger partial path planning  # Line 79
```

**Implementation Detail**: The function prioritizes points closer to `belief.mean` (minimizing `distances_to_mean`, line 64). This is a deviation from pure Boltzmann sampling and represents a greedy heuristic.

---

## 4. Cost Calculation: Three-Term Objective

**File**: `src/samplers/adaptive_goal_mcmc.py`

### Cost Function (`_compute_cost` method, lines 190-229)

```python
J(q) = λ_goal * J_goal(q) + λ_clearance * J_clearance(q) + info_gain_cost
```

**Boundary/Collision Handling** (lines 204-208):
```python
if not np.all(bounds[0] <= q) or not np.all(q <= bounds[1]):
    return _INFINITE_COST  # 1e10, line 15

if obstacles is not None and not point_is_free(q, obstacles):
    return _INFINITE_COST
```

**Term 1: Goal Proximity** (lines 210-214):
```python
if goal_belief is not None:
    goal_cost = self.lambda_goal * goal_belief.expected_distance(q)  # Line 211
else:
    center = (bounds[0] + bounds[1]) / 2.0
    goal_cost = self.lambda_goal * np.linalg.norm(q - center)  # Line 214
```

Where `expected_distance(q)` returns `np.linalg.norm(q - self.mean)` (goal_belief.py, line 101).

**Term 2: Obstacle Clearance** (lines 216-220):
```python
if obstacles is not None:
    clearance = self._compute_clearance(q, obstacles)                           # Line 217
    clearance_cost = self.lambda_clearance / max(clearance, self.clearance_epsilon)  # Line 218
else:
    clearance_cost = 0.0
```

Where `_compute_clearance` returns minimum distance to obstacle boundaries (lines 231-240).

**Term 3: Information Gain** (lines 222-226):
```python
info_gain_cost = 0.0
if goal_belief is not None and sensor_config is not None:
    info_gain = goal_belief.predict_information_gain(q, sensor_config)  # Line 225
    info_gain_cost = -self.lambda_info * info_gain  # Negative = reward  # Line 226
```

**Final Cost** (lines 228-229):
```python
total_cost = goal_cost + clearance_cost + info_gain_cost  # Line 228
return max(total_cost, 0.0)  # Clamp to non-negative
```

**Default Weights** (lines 30-34):
- `lambda_goal = 1.0`
- `lambda_clearance = 0.5`
- `lambda_info = 2.0`

**Implementation Detail**: The information gain term uses a **negative sign** (line 226), converting maximization of information gain into minimization of cost. This is added directly to the cost rather than being a separate reward channel.

---

## 5. MCMC Sampling: Metropolis-Hastings

**File**: `src/samplers/adaptive_goal_mcmc.py`

### Acceptance Probability (`_compute_log_acceptance` method, lines 153-173)

Standard Metropolis-Hastings log-acceptance ratio:

```python
log_pi_proposed = self._log_target_density(q_proposed, ...)  # Line 166
log_pi_current = self._log_target_density(q_current, ...)    # Line 169

return min(0.0, log_pi_proposed - log_pi_current)  # Line 173
```

Where the target density is:

```python
def _log_target_density(...) -> float:
    cost = self._compute_cost(q, ...)
    return -cost / self.temperature  # Line 188
```

So: `log π(q) = -J(q) / T` where `T = self.temperature` (default: 10.0, line 29).

### Accept/Reject Decision (`_accept_proposal` method, lines 242-252)

```python
if log_acceptance >= 0.0:
    return True  # Always accept if proposed state is better  # Line 249

acceptance_prob = np.exp(log_acceptance)  # Line 251
return self.rng.random() < acceptance_prob  # Line 252
```

### Proposal Distribution (`_propose_new_state` method, lines 146-151)

Gaussian random walk with clamping to bounds:

```python
noise = self.rng.normal(0.0, self.proposal_std, size=self._current_state.shape)  # Line 148
q_proposed = self._current_state + noise                                         # Line 149
q_proposed = np.clip(q_proposed, bounds[0], bounds[1])                           # Line 150
return q_proposed
```

**Default `proposal_std = 5.0`** (line 28).

**Implementation Detail**: The proposal is symmetric Gaussian, but clamping to bounds introduces asymmetry near boundaries. The implementation does **NOT** apply boundary correction terms to the acceptance ratio, which technically violates detailed balance at the boundary.

---

## 6. Lambda Mixing: Uniform vs MCMC Hybrid

**File**: `src/samplers/adaptive_goal_mcmc.py`

### Sampling Strategy (`sample` method, lines 61-111)

The sampler uses a **λ-biased hybrid strategy**:

```python
# Lambda-biasing: probabilistically choose between uniform and MCMC sampling
r = self.rng.random()  # Line 92
if r < self.uniform_mixing_rate:  # Line 93
    # Uniform exploration mode
    # MCMC chain state persists (is NOT reset)
    return self._sample_uniform(bounds, obstacles)  # Line 96

# MCMC exploitation mode: perform Metropolis-within-Gibbs step
if self._current_state is None:  # Line 99
    self._current_state = self._sample_uniform_initial(bounds, obstacles)  # Line 100
    return self._current_state.copy()

q_proposed = self._propose_new_state(bounds)  # Line 103
log_acceptance = self._compute_log_acceptance(...)  # Line 104

if self._accept_proposal(log_acceptance):  # Line 108
    self._current_state = q_proposed  # Line 109

return self._current_state.copy()  # Line 111
```

**Key Variables**:
- `self.uniform_mixing_rate`: Probability λ of selecting uniform sample (default: 0.1, line 33)
- `self._current_state`: Persistent MCMC chain state (lines 58, 100, 109, 111)

**Implementation Detail**: When a uniform sample is selected, the MCMC chain state is **NOT reset** (line 95 comment). The chain "pauses" and resumes on the next MCMC iteration. This is a deviation from standard mixture distributions where components are typically independent.

---

## 7. Robust Planning: Partial Path Fallback

**File**: `src/rrt/planner.py`

### Best Node Tracking (`plan` method, lines 105-178)

The planner tracks the best node during tree expansion:

```python
# Initialize tree with start position
tree = Tree(self.start)  # Line 120

# Track the best node (closest to goal) for fallback
best_node_idx = 0  # Start node  # Line 123
best_distance_to_goal = np.linalg.norm(self.start - self.goal)  # Line 124

# Main planning loop
for iteration in range(self.max_iterations):  # Line 127
    # ... sampling, steering, collision checking ...

    if segment_is_free(q_near, q_new, self.obstacles):  # Line 151
        new_idx = tree.add_node(nearest_idx, q_new)  # Line 153

        # Update best node tracker
        distance_to_goal = np.linalg.norm(q_new - self.goal)  # Line 156
        if distance_to_goal < best_distance_to_goal:  # Line 157
            best_node_idx = new_idx  # Line 158
            best_distance_to_goal = distance_to_goal  # Line 159

        # Check if we can reach the goal...
        if is_goal_reached(q_new, self.goal, self.goal_threshold):  # Line 163
            if segment_is_free(q_new, self.goal, self.obstacles):  # Line 165
                goal_idx = tree.add_node(new_idx, self.goal)  # Line 167
                path = self._extract_path(tree, goal_idx)  # Line 168
                return path  # SUCCESS: Complete path found  # Line 169
```

### Partial Path Fallback (lines 172-178)

If max iterations exceeded without reaching goal:

```python
# Failed to find complete path within max_iterations
if return_partial:  # Line 173
    # Return partial path to best node (closest to goal)
    partial_path = self._extract_path(tree, best_node_idx)  # Line 175
    return partial_path
else:
    return None  # Line 178
```

**Best Node Definition**: The node with minimum Euclidean distance to `self.goal` (line 156-159).

**Default Behavior**: `return_partial=True` is the default (line 105), so the planner **always returns a path** unless explicitly configured otherwise.

**Implementation Detail**: The "best" node is defined purely by distance to goal (line 157), **NOT** by information gain or any other heuristic. This contrasts with the documentation comment which mentions "highest heuristic score (closest to target or highest information gain)" - only distance is actually implemented.

---

## 8. Execution Loop: Replanning Triggers

**File**: `src/rrt/navigation.py`

### Main Navigation Loop (`run_closed_loop_navigation` function)

The execution loop has **two distinct conditions** that trigger replanning:

### Condition 1: Path Exhaustion (lines 174-177)

```python
needs_replan = (
    current_path is None  # Initial planning  # Line 175
    or len(current_path) <= 1  # Exhausted current path  # Line 176
)
```

Triggers when:
- `current_path is None`: Initial planning (start of navigation)
- `len(current_path) <= 1`: Current path has 0 or 1 waypoints remaining

### Condition 2: Event-Triggered Replanning (lines 248-252)

```python
entropy_before_execution = current_belief.compute_entropy()  # Line 232

# ... (inside execution loop, line 234) ...
# Collect sensor observation at new position
observation, obs_covariance = sensor.observe(current_pose, true_goal_array)  # Line 243
current_belief.update(observation, obs_covariance)  # Line 246

# Check for event-triggered replanning with hysteresis
entropy_after_observation = current_belief.compute_entropy()  # Line 249
entropy_reduction = entropy_before_execution - entropy_after_observation  # Line 250

if entropy_reduction >= entropy_threshold and steps_since_last_replan >= min_steps_between_replans:  # Line 252
    # Trigger early replanning
    ...
```

Triggers when **BOTH** conditions are met:
1. `entropy_reduction >= entropy_threshold`: Significant information gain from observations
2. `steps_since_last_replan >= min_steps_between_replans`: Hysteresis condition to prevent thrashing

**Default Parameters**:
- `entropy_threshold = 0.1` (line 95)
- `min_steps_between_replans = 5` (line 97)
- `execution_horizon = 10` (line 94)

**Entropy Metric**: `compute_entropy()` returns `trace(self.covariance)` (goal_belief.py, line 162), a proxy for Gaussian entropy.

### Replanning Logic (lines 253-306)

When event-triggered replanning occurs:

```python
# Get valid target from updated belief
replan_target = get_valid_target(current_belief, obstacles, bounds, num_samples=100)  # Line 258

# Try to replan immediately (Safety Net: don't clear old path yet)
planner = RRTPlanner(...)  # Line 261
new_path = planner.plan(return_partial=True)  # Line 274

if new_path is not None:
    # Replanning succeeded (or returned partial)! Use new path
    num_replans += 1  # Line 286
    steps_since_last_replan = 0  # Line 287

    # Remove current position from new path
    if len(new_path) > 0 and np.allclose(new_path[0], current_pose):  # Line 290
        new_path = new_path[1:]

    # Replace current path with new path
    if len(new_path) > 0:  # Line 295
        current_path = new_path  # Line 296

    break  # Start executing new path  # Line 301
else:
    # Replanning failed catastrophically - Safety Net: continue with old path
    print(f"⚠️  Warning: Event-triggered replanning failed...")  # Line 304
    # Continue executing remaining waypoints from current_path
```

**Implementation Detail**: The event-triggered replan does **NOT** clear the old path before attempting to replan (line 260 comment). If replanning fails (`new_path is None`), the robot continues executing the old path. This is a safety mechanism to prevent the robot from getting stuck.

---

## Summary of Key Implementation Deviations

1. **Kalman Update**: Uses `(I - K)Σ` form instead of symmetrized Joseph form
2. **Target Selection**: Greedy selection of valid sample closest to mean (not Boltzmann sampling)
3. **Best Node Definition**: Uses only distance to goal, not information gain or other heuristics despite documentation suggesting otherwise
4. **MCMC Boundary Correction**: No correction terms for clamped proposals at workspace boundaries
5. **Lambda Mixing Chain Persistence**: MCMC chain state persists when uniform samples are selected (non-standard mixture behavior)
6. **Information Gain as Cost**: Information gain is negated and added to cost function, not treated as separate reward

---

## File Cross-Reference

| Section | Primary File | Key Lines |
|---------|-------------|-----------|
| Sensing | `src/rrt/goal_sensor.py` | 46-79 |
| Belief Update | `src/rrt/goal_belief.py` | 46-71 |
| Target Selection | `src/rrt/navigation.py` | 21-79, 179-187 |
| Cost Calculation | `src/samplers/adaptive_goal_mcmc.py` | 190-229 |
| MCMC Sampling | `src/samplers/adaptive_goal_mcmc.py` | 153-173, 242-252 |
| Lambda Mixing | `src/samplers/adaptive_goal_mcmc.py` | 61-111 |
| Robust Planning | `src/rrt/planner.py` | 105-178 |
| Execution Loop | `src/rrt/navigation.py` | 82-343 |

---

**Document Status**: This specification reflects the exact implementation as of the current codebase state. All line numbers and variable names are accurate to the source files.
