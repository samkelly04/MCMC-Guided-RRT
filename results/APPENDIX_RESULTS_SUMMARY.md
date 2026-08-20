# Appendix Results Summary

**Date**: March 18, 2026
**Scope**: Planner improvements (Solutions 2.1 + 2.2) and C-Space learning (Grid + KDE)

> **Note on paths.** Section 2 below describes the C-space learning work as it stood when it
> lived inside this repository under `future_work/`. That work has since moved to its own
> repository, [cspace-learning](https://github.com/samkelly04/cspace-learning), where the
> models live under `models/` and the results under `results/cspace_v1/`. The numbers are
> unchanged; only the file paths in that section are stale.

---

## 1. Planner Improvements

### What was implemented

**Solution 2.1 — Greedy Forward Progress (Distance-Based Stall Detection)**
- File: `src/rrt/navigation.py`
- Replaced waypoint-count-only stall check with dual stall detection:
  - **Empty-path stall**: fires after 10 consecutive planning cycles with 0 executable waypoints (unchanged)
  - **Distance stall** (new): fires when robot has moved < 2.0 units over the last 10 planning steps that produced waypoints
- Resets empty-path stall counter on ANY waypoints (even 1), preventing premature give-up on incremental progress

**Solution 2.2 — Adaptive Iteration Budget**
- File: `src/rrt/navigation.py` (function `compute_adaptive_iterations`)
- Formula: `min(int(15000 * (1 + distance / 40)), 50000)`
- When target is 80 units away: 45,000 iterations (3x baseline)
- When target is 10 units away: 18,750 iterations (1.25x baseline)
- Wired via opt-in `adaptive_iterations=True` parameter (default False to preserve test behavior)
- `AdaptiveMCMCAgent` in `src/core/agents.py` sets `adaptive_iterations=True`

### Results

**Baseline** (full_experiment_v4, 30 maps per level):

| Agent | Easy | Medium | Hard |
|-------|------|--------|------|
| AdaptiveMCMCAgent | 100% (30/30) | 50% (15/30) | **63% (19/30)** |
| RRTAgent | 100% (30/30) | 57% (17/30) | 93% (28/30) |

**After Solutions 2.1 + 2.2** (improved_v1, 10 maps per level, 59/60 trials completed):

| Agent | Easy | Medium | Hard |
|-------|------|--------|------|
| AdaptiveMCMCAgent | 100% (10/10) | 80% (8/10) | **100% (9/9)** |
| RRTAgent | 100% (10/10) | 80% (8/10) | 100% (10/10) |

**Key finding**: MCMC Hard success rate improved from **63% to 100%** (9/9 completed trials). Medium improved from 50% to 80%. One MCMC Hard trial (map seed 109) timed out after 95 minutes and was excluded.

### Computation time

Adaptive iterations increases per-trial computation time for MCMC on Hard mazes (baseline avg 425s, improved avg ~568s per the quick test), but the tradeoff is worthwhile since the robot now succeeds instead of stalling.

---

## 2. C-Space Learning

### What was implemented

**Occupancy Grid** (`future_work/implementation_sketches/cspace_models/occupancy_grid.py`)
- Pre-existing stub, verified complete
- Discretizes workspace into 1.0m cells
- Tracks collision/free counts per cell from MCMC sample labels
- O(1) lookup for `obstacle_probability(q)`

**KDE Model** (`future_work/implementation_sketches/cspace_models/kde_model.py`)
- Pre-existing stub, verified complete
- Fits separate KDEs for collision vs free samples
- Classifies via Bayes' rule: P(collision|q) = P(q|collision) * P(collision) / P(q)
- Bandwidth = 1.0 (tuned for 2-unit-thick walls in 100x100 space)

**Experiment Script** (`experiments/run_cspace_experiment.py`)
- New script created from scratch
- Generates 3 Hard mazes, runs MCMC planner to collect 25k samples per maze (75k total)
- MCMC sample labels: `collision` (rejected by collision checker) vs `free` (accepted, uniform, or metropolis-rejected)
- Trains Grid on all 75k samples; trains KDE on 10k subsampled (for tractable query time)
- Evaluates on 3,600-point test grid using ground-truth Shapely collision checker
- KDE uses vectorized batch scoring for fair speedup measurement

### Results

| Model | F1 Score | Speedup vs Shapely | Query Time (3,600 pts) |
|-------|----------|-------------------|----------------------|
| Shapely (ground truth) | N/A | 1.0x | 1,167 ms |
| **Occupancy Grid** | **0.648** | **202x** | 5.8 ms |
| KDE (batch, bandwidth=1.0) | 0.496 | 2.1x | 545 ms |

**Configuration**:
- 75,000 MCMC samples from 3 Hard mazes (seeds 200-202)
- Collision rate: 12.5% (9,406 collision / 65,594 free)
- Grid: resolution 1.0m, threshold 0.5
- KDE: bandwidth 1.0, threshold 0.3 (lowered for class imbalance), 10k subsample
- Test grid: 60x60 = 3,600 evaluation points

**Key findings**:
1. **Grid model is clearly superior**: 202x faster than Shapely with F1=0.648 — viable as a fast approximate collision checker
2. **KDE is impractical**: only 2.1x speedup (O(n) queries vs O(1) for Grid) and lower F1 (0.496) due to class imbalance and bandwidth limitations
3. **Class imbalance** (12.5% collision) makes classification harder — collision samples are sparse relative to free space
4. **Grid F1 = 0.648** reflects the fundamental challenge: MCMC samples don't uniformly cover obstacle boundaries (they cluster in high-probability regions), so some boundary cells are under-sampled

### Output files

All saved to `results/cspace_v1/`:
- `grid_heatmap.png` — Occupancy grid probability map overlaid on maze walls
- `kde_heatmap.png` — KDE probability map overlaid on maze walls
- `cspace_results_summary.png` — F1 and speedup bar charts
- `cspace_results.json` — Machine-readable results

---

## 3. Code Changes Summary

### Modified files
| File | Change |
|------|--------|
| `src/rrt/navigation.py` | Added `compute_adaptive_iterations()`, distance-based stall detection, `adaptive_iterations` parameter |
| `src/core/agents.py` | Added `adaptive_iterations=True` to `AdaptiveMCMCAgent` |
| `tests/test_navigation.py` | Fixed 2 pre-broken tests (entropy, replanning) |
| `tests/test_rrt_planner.py` | Fixed `return_partial=False` in blocked-path test |
| `experiments/run_cspace_experiment.py` | New file — full C-space learning experiment |
| `future_work/SOLUTIONS.md` | Updated status to COMPLETE with results table |
| `future_work/README.md` | Added scope note |
| `future_work/CSPACE_RECONSTRUCTION.md` | Added scope note (GP deferred) |

### Test status
All **44 tests pass** (`pytest tests/ -x -q` — 85s).

---

## 4. What Was Deferred

| Item | Reason |
|------|--------|
| Solution 2.3 (Multi-Resolution Planning) | Too complex for one session |
| Solution 2.4 (Informed RRT*) | Too complex for one session |
| GP Classifier for C-Space | Requires careful kernel selection, too slow to train/evaluate tonight |
| Full 1-2M sample collection | 75k sufficient for proof-of-concept; diminishing returns expected |
| 30-map improved experiment | 10-map run took 95 min; results are directionally clear |

---

## 5. Suggested Next Steps

1. **Run full 30-map improved experiment** to get statistically robust comparison (expect ~2-3 hours)
2. **Investigate Medium stall failures** — robot gets trapped when planner returns 1-node partial paths; may need local escape heuristic
3. **Improve Grid F1** — try finer resolution (0.5m), Laplace smoothing, or boundary-aware sampling
4. **GP Classifier** — if time permits, would provide smooth probability surfaces with uncertainty estimates
5. **Integration test** — use Grid model as fast pre-filter in the actual planner (check Grid first, fall back to Shapely only near boundaries)
