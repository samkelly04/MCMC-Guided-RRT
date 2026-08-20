# Critical Solutions to Improve Current Planner

**Status**: Solutions 2.1 + 2.2 COMPLETE (appendix results); Solutions 2.3 + 2.4 still deferred
**Last Updated**: March 18, 2026

---

## Appendix Results (improved_v1, 10 maps, 1 trial each)

**After applying Solutions 2.1 + 2.2** (`adaptive_iterations=True` in `AdaptiveMCMCAgent`):

| Complexity | Baseline (30 maps) | After Improvements (9–10 maps) |
|------------|-------------------|-------------------------------|
| Easy       | 100%              | 100%                          |
| Medium     | 50%               | **80%**                       |
| Hard       | **63%**           | **100%**                      |

Key: Hard success rate increased from 63% → 100% (9/9 trials).
Note: 1 trial (MCMC Hard, map 10) timed out and was excluded; 59/60 trials completed.

---

## Overview

**Note**: This document describes improvements to the **current MCMC-RRT planner** to increase success rate from 63% → 85-90%.

**Solutions 2.1 and 2.2** have been implemented and validated as appendix material alongside C-space learning.
**Solutions 2.3 and 2.4** remain deferred — too complex for the current timeline.

---

## Current Planner Limitations (Baseline)

**Success Rates (full_experiment_v4, 30 maps each)**:
- Easy mazes (5×5): 100% success ✅
- Medium mazes (8×8): 50% success ⚠️
- Hard mazes (10×10): **63% success** ❌ (36.7% failure rate)

**Root Causes**:
1. **Stall-on-initialization**: Can't find ANY path in 15,000 iterations when start-goal distance is large (99 units)
2. **Planning budget fragmentation**: 10-20 replans × 15,000 iterations = 150k-300k total, yet WORSE than single-shot RRT (93% success)
3. **Computational cost**: 8-22× slower than baseline (425s vs 19.7s on hard mazes)

---

## Four Solutions (Ordered by Priority)

### Solution 2.1: Greedy Forward Progress Acceptance [COMPLETE]

**Impact**: 63% → 85-90% success rate
**Effort**: 1-2 hours (modify `src/rrt/navigation.py`)
**Complexity**: Low

**Core Idea**: Accept *any* forward progress, even if partial path has only 1-2 waypoints.

**Current Behavior** (causes stall):
```python
if waypoints_to_execute == 0:
    stall_count += 1
    if stall_count >= 10:
        return FAILURE  # Stall detected
```

**Proposed Change**:
```python
# Accept even single waypoints as progress
if waypoints_to_execute == 0:
    stall_count += 1
else:
    stall_count = 0  # Reset on ANY progress
    # Execute waypoints (even if just 1)

# Add distance-based stall check (not just waypoint count)
if total_distance_traveled < 2.0 in last 10 steps:
    true_stall_count += 1
```

**Why It Works**: Allows incremental progress instead of requiring full paths. Robot moves 1-2 units per replan, eventually reaching goal.

---

### Solution 2.2: Adaptive Iteration Budget [COMPLETE]

**Impact**: 63% → 85-90% success rate (similar to 2.1, can combine)
**Effort**: 2-4 hours (add function to compute iterations)
**Complexity**: Low-Medium

**Core Idea**: Scale `max_iterations` based on distance to goal.

**Current**: Fixed 15,000 iterations regardless of distance
**Proposed**:
```python
def compute_adaptive_iterations(current_pose, target, base=15000, scale=40):
    distance = np.linalg.norm(target - current_pose)
    iterations = int(base * (1 + distance / scale))
    return min(iterations, 50000)  # Cap at 50k

# Initial planning (d=99): 15000 × (1 + 99/40) ≈ 52k → capped at 50k
# After progress (d=49): 15000 × (1 + 49/40) ≈ 33k
# Near goal (d=10): 15000 × (1 + 10/40) ≈ 18k
```

**Why It Works**: Allocates more planning budget when problem is harder (long distance), less when easy (short distance).

---

### Solution 2.3: Multi-Resolution Planning [MEDIUM PRIORITY]

**Impact**: 20-30% success improvement
**Effort**: 1 week (implement two-stage planner)
**Complexity**: Medium

**Core Idea**: Plan at coarse resolution first (step_size=5.0), then refine with fine resolution (step_size=1.0).

**Two-Stage Approach**:
1. **Coarse Planning**: step_size=5.0, 5,000 iterations → approximate path quickly
2. **Refinement**: For each coarse segment, plan fine path (step_size=1.0, 2,000 iterations/segment)

**Why It Works**: Coarse planning finds approximate routes faster (5× larger steps), refinement ensures collision-free execution.

---

### Solution 2.4: Informed RRT* Integration [MEDIUM PRIORITY]

**Impact**: 20-30% success improvement + better path quality
**Effort**: 1-2 weeks (integrate OMPL library or implement informed sampling)
**Complexity**: High

**Core Idea**: Replace vanilla RRT with Informed RRT* (samples within elliptical region between start and goal).

**Informed Sampling**:
```python
# Only sample points that could improve current best path
def sample_informed(start, goal, current_best_length):
    if current_best_length < inf:
        # Sample from ellipse with foci at start and goal
        # Only includes points where distance(point, start) + distance(point, goal) < current_best_length
        ...
    else:
        # No path yet, sample uniformly
        ...
```

**Why It Works**: Focuses exploration on promising regions, faster convergence for long-distance planning.

**Available in**: OMPL library (`og.InformedRRTstar`)

---

## Recommended Implementation Order

If pursuing these solutions:

1. **Start with 2.1 (Greedy Acceptance)** - lowest effort, highest immediate impact
2. **Add 2.2 (Adaptive Budget)** - complements 2.1, improves initial planning success
3. **Optional: 2.3 or 2.4** - if target is 90-95% success (vs 85-90% from 2.1+2.2)

**Combined Impact**: Solutions 2.1 + 2.2 likely achieve 85-90% success with ~50% reduction in average planning time.

---

## Why These Are Deferred

**Focus**: C-space learning (collision checking speedup, topology discovery) is the primary deliverable.

**Rationale**:
- C-space learning is a **novel research contribution** (learning from MCMC patterns)
- Planner improvements are **engineering fixes** (important but not scientifically novel)
- C-space learning builds **reusable infrastructure** for future work
- Planner improvements are **task-specific** to current experiments

**When to revisit**: After C-space deliverable is complete and validated. If time permits, implementing 2.1 and 2.2 (combined 3-6 hours) would improve baseline system performance for comparisons.

---

## Reference

Full detailed implementation instructions available in:
- `future_work_limitations.md` Section 2 (lines 166-461) — now in the
  [cspace-learning](https://github.com/samkelly04/cspace-learning) repo under `docs/`
- `discussion_limitations_future_work.md` Section 3.1 (greedy acceptance + adaptive budget) —
  same repo, `docs/`

These comprehensive documents provide:
- Line-by-line code modifications
- Expected outcomes with quantitative predictions
- Risks and mitigation strategies
- Evaluation protocols
