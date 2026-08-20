# Experimental Design

## Overall Design

We compare two path planning approaches across maze environments of varying complexity:

1. **Uniform RRT Baseline**: Standard RRT with uniform random sampling and perfect goal knowledge
2. **Adaptive MCMC-RRT**: Receding horizon navigation with MCMC-guided sampling, uncertain goal localization, and event-triggered replanning

For each of three complexity levels (Easy: 6×6 grid, Medium: 8×8 grid, Hard: 10×10 grid), we generate 30 unique maze instances using random seeds 100-129. Each maze is created with fixed start and goal positions determined by the seed, ensuring both agents solve identical problem instances. This paired experimental design enables direct statistical comparison of performance metrics.

Both agents use identical RRT parameters (step size: 5.0, goal threshold: 5.0, max iterations per planning call: 10,000-15,000) to isolate the effect of the sampling strategy. The Adaptive MCMC agent begins with high goal uncertainty (initial belief: N([50,50], 400I)) and refines its estimate through distance-dependent sensor observations (σ_far = 10.0, σ_near = 0.5) during navigation. Replanning is triggered when entropy reduction exceeds a threshold (default: 5.0) with hysteresis to prevent thrashing (minimum 3 steps between replans).

This yields 180 total trials (30 maps × 3 complexity levels × 2 agents), providing sufficient statistical power for hypothesis testing while maintaining computational feasibility.

## Evaluation Metrics

### Primary Performance Metrics
- **Success Rate**: Proportion of trials reaching the goal within step/iteration limits
- **Computation Time**: Wall-clock time from planning initialization to solution (seconds)
- **Path Length**: Total Euclidean distance traveled along executed trajectory

### Adaptive Navigation Metrics (MCMC-RRT only)
- **Number of Replans**: Count of replanning events triggered during execution
- **Belief Convergence**: Final distance between goal belief mean and true goal location (accuracy metric)
- **Final Belief Entropy**: trace(Σ_final) quantifying remaining uncertainty about goal
- **Belief Evolution**: Full trajectory of goal belief means with uncertainty ellipses over execution

### Statistical Analysis
All pairwise agent comparisons employ:
- **Mann-Whitney U test**: Non-parametric test for differences in central tendency
- **Independent samples t-test**: Parametric test (when normality assumptions hold)
- **Cohen's d effect size**: Quantifies practical significance of observed differences
- **95% Confidence intervals**: For all mean metrics

Results are stratified by complexity level to assess how performance scaling differs between approaches. Significance threshold: α = 0.05 with Bonferroni correction for multiple comparisons.

### Visualization Suite
- **Comparative bar charts**: Success rates by agent and complexity
- **Box plots**: Computation time and path length distributions
- **Belief evolution plots**: Spatial trajectory of belief mean with 2-sigma uncertainty ellipses
- **Entropy analysis**: Temporal plot of trace(Σ) with replan event markers
- **Replan frequency**: Distribution of replanning counts across trials
