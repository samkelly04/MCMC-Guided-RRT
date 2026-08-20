# Prompt: Integrate Goal Belief and Sensor Config with RRT Planner

## Objective
Modify the `RRTPlanner` class to support adaptive goal-directed planning by passing goal belief and sensor configuration to the sampler. This enables the MCMC sampler to use information-theoretic cost functions and active perception strategies.

## Background
Currently, the planner calls `sampler.sample(bounds)` without passing any state information. The `AdaptiveGoalMCMCSampler` expects a `state` dict containing `goal_belief`, `obstacles`, and `sensor_config` to compute information gain and expected distance costs. This integration is required before implementing receding-horizon navigation.

## Changes Required

### 1. Add Entropy Computation Helper to GoalBelief

**File**: `src/rrt/goal_belief.py`

Add a method to compute entropy (uncertainty proxy) from the covariance matrix:

```python
def compute_entropy(self) -> float:
    """Compute entropy proxy using trace of covariance matrix.
    
    For a Gaussian distribution N(μ, Σ), entropy is proportional to 
    log(det(Σ)). We use trace(Σ) as a simpler proxy that captures
    the total uncertainty across all dimensions.
    
    Returns:
        Entropy proxy: trace(covariance)
    """
    return float(np.trace(self.covariance))
```

**Location**: Add this method after `predict_information_gain()` method (around line 136).

**Rationale**: Needed for tracking uncertainty reduction in receding-horizon loops and event-triggered replanning.

---

### 2. Expose Sensor Config from GoalObservationModel

**File**: `src/rrt/goal_sensor.py`

Add a property to expose the sensor configuration:

```python
@property
def config(self) -> GoalSensorConfig:
    """Return the sensor configuration."""
    return self._config
```

**Location**: Add this property after `__init__` method (around line 40).

**Rationale**: Allows external code to extract sensor config for passing to samplers without accessing private `_config` attribute.

---

### 3. Modify RRTPlanner to Accept Optional Goal Belief and Sensor Config

**File**: `src/rrt/planner.py`

#### 3.1 Update `__init__` Method

Add optional parameters for goal belief and sensor config:

```python
def __init__(
    self,
    start: np.ndarray,
    goal: np.ndarray,
    obstacles: List[Polygon],
    sampler: BaseSampler,
    step_size: float,
    goal_threshold: float,
    bounds: np.ndarray,
    max_iterations: int = 10000,
    goal_belief: Optional[GoalBelief] = None,  # NEW
    sensor_config: Optional[GoalSensorConfig] = None,  # NEW
):
```

**Changes**:
- Import `Optional` from `typing` if not already imported
- Import `GoalBelief` from `.goal_belief`
- Import `GoalSensorConfig` from `.goal_sensor`
- Store as instance variables: `self.goal_belief = goal_belief` and `self.sensor_config = sensor_config`
- **Important**: If `goal_belief` is provided, use `goal_belief.mean` as the planning target instead of `self.goal`. Update the goal assignment:
  ```python
  if goal_belief is not None:
      self.goal = np.array(goal_belief.mean, dtype=float)
  else:
      self.goal = np.array(goal, dtype=float)
  ```

**Rationale**: Allows planner to plan toward belief mean (adaptive) while maintaining backward compatibility with fixed goals.

#### 3.2 Update `plan()` Method to Pass State to Sampler

Modify line 103 where `sampler.sample()` is called:

**Current code**:
```python
q_rand = self.sampler.sample(self.bounds)
```

**New code**:
```python
# Construct state dict for adaptive samplers
state = None
if self.goal_belief is not None or self.sensor_config is not None:
    state = {}
    if self.goal_belief is not None:
        state['goal_belief'] = self.goal_belief
    if self.obstacles is not None:
        state['obstacles'] = self.obstacles
    if self.sensor_config is not None:
        state['sensor_config'] = self.sensor_config

q_rand = self.sampler.sample(self.bounds, state=state)
```

**Rationale**: Passes necessary context to adaptive samplers while maintaining backward compatibility (uniform samplers ignore state).

#### 3.3 Update Validation Logic

Modify `_validate_inputs()` to handle goal_belief:

**Current code** (line 86-87):
```python
if not np.all(self.bounds[0] <= self.goal) or not np.all(self.goal <= self.bounds[1]):
    raise ValueError("Goal position is outside bounds")
```

**New code**:
```python
# Validate goal (either from parameter or goal_belief.mean)
goal_to_check = self.goal
if self.goal_belief is not None:
    goal_to_check = self.goal_belief.mean
    
if not np.all(self.bounds[0] <= goal_to_check) or not np.all(goal_to_check <= self.bounds[1]):
    raise ValueError("Goal position is outside bounds")
```

**Note**: Actually, since we already set `self.goal = goal_belief.mean` in `__init__`, the validation can remain as-is. But add a comment explaining this.

---

### 4. Update Type Hints and Imports

**File**: `src/rrt/planner.py`

Add imports at the top:
```python
from typing import List, Optional  # Add Optional if not present
from .goal_belief import GoalBelief
from .goal_sensor import GoalSensorConfig
```

---

## Testing Requirements

After implementing these changes, verify:

1. **Backward Compatibility**: Existing code that creates `RRTPlanner` without `goal_belief` or `sensor_config` should still work.

2. **State Passing**: When `goal_belief` and `sensor_config` are provided, verify that:
   - The sampler receives them in the state dict
   - The planner uses `goal_belief.mean` as the planning target
   - MCMC sampler can compute information gain

3. **Edge Cases**:
   - `goal_belief=None, sensor_config=None`: Should work (backward compatible)
   - `goal_belief=some_belief, sensor_config=None`: Should pass belief to sampler
   - `goal_belief=None, sensor_config=some_config`: Should pass config to sampler
   - Both provided: Should pass both

## Implementation Order

1. Add `compute_entropy()` to `GoalBelief` (simple addition)
2. Add `config` property to `GoalObservationModel` (simple addition)
3. Modify `RRTPlanner.__init__()` to accept and store optional parameters
4. Update goal assignment logic in `__init__()` to use `goal_belief.mean` if provided
5. Modify `plan()` method to construct and pass state dict to sampler
6. Update imports and type hints

## Expected Behavior After Changes

- **Without goal_belief/sensor_config**: Planner works exactly as before (backward compatible)
- **With goal_belief**: Planner plans toward `goal_belief.mean` instead of fixed goal
- **With sensor_config**: Sampler can compute information gain for active perception
- **With both**: Full adaptive planning with information-theoretic cost function

## Notes

- The `goal_belief` parameter is optional to maintain backward compatibility
- When `goal_belief` is provided, `goal` parameter is still required but may be ignored (we use `goal_belief.mean` instead)
- Consider adding a docstring note explaining that when `goal_belief` is provided, the `goal` parameter is overridden by `goal_belief.mean`
- The state dict construction handles `None` values gracefully - only includes keys that are not `None`
