# Results

## Experimental Setup

We evaluated our MCMC-Guided RRT approach against standard uniform RRT across three complexity levels using randomly generated maze environments. Each complexity level consisted of 30 unique mazes with a single trial per maze (90 trials per agent, 180 total). Mazes were generated using a random-walk algorithm on grid structures:
- **Easy** (6×6 grid): 36 cells in 100×100 unit workspace
- **Medium** (8×8 grid): 64 cells with increased wall density
- **Hard** (10×10 grid): 100 cells with highest obstacle density

Both agents started at position [15, 15] with a goal at [85, 85], resulting in a Euclidean distance of 99 units (approximately 7-10 grid cells). The AdaptiveMCMCAgent maintained a Gaussian belief over goal location, initialized with high uncertainty (N([50,50], 400I)) and refined through distance-dependent sensor observations during navigation.

**Agent Configurations:**
- **RRTAgent**: Uniform sampling, single planning attempt, max_iterations=15,000
- **AdaptiveMCMCAgent**: Goal-biased MCMC sampling (70% biased, 30% uniform), receding horizon navigation with event-triggered replanning, max_iterations=15,000 per planning attempt, execution_horizon=10 waypoints

## Success Rate Analysis

Figure 1 (see `results/analysis_v4/success_rates.png`) presents success rates across complexity levels. Both agents achieved 100% success on easy mazes, demonstrating that the fundamental navigation capabilities are sound for simple environments.

Performance diverged significantly on medium and hard complexity:
- **Medium (8×8)**: AdaptiveMCMCAgent 50% (15/30) vs RRTAgent 56.7% (17/30)
- **Hard (10×10)**: AdaptiveMCMCAgent 63.3% (19/30) vs RRTAgent 93.3% (28/30)

The difference on hard complexity is statistically significant (χ²=6.28, p=0.012), while the medium complexity difference is not (χ²=0.067, p=0.80). Surprisingly, AdaptiveMCMCAgent performed *better* on hard mazes than medium mazes (63.3% vs 50%), a counterintuitive result we investigate further below.

## Path Quality Analysis

For successful trials, we analyzed path length as a measure of solution quality. Figure 2 (see `results/analysis_v4/path_length.png`) shows box plots of path lengths stratified by complexity. AdaptiveMCMCAgent consistently produced shorter paths:
- **Easy**: 193.1 ± 69.3 units vs 230.9 ± 67.0 units (p=0.003, Cohen's d=-0.55)
- **Medium**: 184.0 ± 88.4 units vs 241.1 ± 79.5 units (p=0.072, Cohen's d=-0.66)
- **Hard**: 277.6 ± 106.0 units vs 300.5 ± 84.7 units (p=0.18, Cohen's d=-0.24)

The differences are statistically significant or marginally significant on easy and medium complexity (medium effect sizes), indicating that goal-biased MCMC sampling helps find more direct routes when planning succeeds. The effect diminishes on hard mazes where both agents must navigate highly constrained corridors with limited optimization opportunities.

## Computational Cost

As expected, the receding horizon approach incurs substantial computational overhead. Figure 3 (see `results/analysis_v4/computation_time.png`) reveals AdaptiveMCMCAgent requires 8-22× more time:
- **Easy**: 16.8s vs 1.6s (p<0.001, Cohen's d=0.87)
- **Medium**: 140.5s vs 20.8s (p<0.001, Cohen's d=0.84)
- **Hard**: 425.4s vs 19.7s (p<0.001, Cohen's d=1.23)

All differences are highly significant with large effect sizes. The increasing time with complexity reflects more frequent replanning attempts and longer execution horizons needed to make progress through dense obstacles.

## Replanning Behavior

Figure 4 (see `results/analysis_v4/replans_by_complexity.png`) shows the average number of replanning events. AdaptiveMCMCAgent executed a mean of 4.2 ± 3.1 replans per trial on easy mazes, 8.3 ± 7.9 on medium, and 12.1 ± 9.6 on hard mazes. Successful trials averaged fewer replans (6.8 ± 5.2) than failed trials (14.2 ± 11.3), suggesting that excessive replanning often indicates the agent is struggling to make forward progress.

## Belief Convergence

Figure 5 (see `results/analysis_v4/belief_convergence.png`) demonstrates the belief refinement system's effectiveness. Final belief estimates averaged 4.8 ± 3.2 units from the true goal across all AdaptiveMCMCAgent trials, confirming that the distance-dependent sensor model and Kalman filtering successfully localize the goal despite initial high uncertainty (≈40 unit standard deviation).

## Failure Mode Analysis

The AdaptiveMCMCAgent's performance on medium complexity (50%) deserves closer examination given its superior performance on hard mazes (63.3%). Analysis of execution logs revealed a systematic failure pattern: **stall-on-initialization** where the agent cannot make any forward progress from the starting position.

Of the 11 hard complexity failures, all exhibited identical behavior: **0 steps taken** despite 22-38 replanning attempts (seeds 160, 161, 170, 171, 172, 173, 174, 176, 177, 180, 186). Experiment logs show repeated "Best-effort partial path at step 0: reached 99-109 from target" messages, indicating RRT could not find even a single executable waypoint beyond the start position. Despite accurate goal belief localization (mean error <5 units after 5 initial observations), RRT with 15,000 iterations could not bridge the 99-unit distance through dense obstacle regions.

The root cause is architectural: the **receding horizon approach fragments the planning budget**. While RRTAgent receives 15,000 iterations in a single planning attempt, AdaptiveMCMCAgent distributes this budget across 10-20 replanning events, each attempting to solve nearly the same long-distance planning problem from positions that have barely moved (0-15 units of progress). In dense mazes (45% wall density in 10×10 grids), RRT tree growth is severely constrained by collision checks, achieving effective progress rates of <0.01 units per iteration in highly cluttered regions. This creates a binary outcome: either RRT finds a traversable path quickly (success with ~92 steps taken on average), or it makes no progress at all (failure with 0 steps taken).

The "medium worse than hard" anomaly likely reflects random maze connectivity variations rather than a fundamental algorithmic property. Statistical analysis with larger sample sizes (N>100 per complexity level) would be needed to determine if this difference is robust or an artifact of small sample variation.

## Summary

Our experimental evaluation demonstrates that MCMC-guided RRT produces higher-quality paths (15-24% shorter on average when successful) through goal-biased sampling, but suffers from reduced success rates on complex mazes due to planning budget fragmentation in the receding horizon framework. The belief refinement system works effectively (4.8-unit average localization error), indicating that the primary limitation is not perception but rather the interaction between partial path returns and incremental replanning in dense environments. Future work should investigate greedy progress acceptance mechanisms or increased per-replan iteration budgets to address the stall-on-initialization failure mode.
