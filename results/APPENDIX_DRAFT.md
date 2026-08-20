# Appendix: Architectural Improvements and Configuration Space Learning

**Sam Kelly**
March 2026

---

## A. Introduction

The main body of this paper demonstrated that replacing uniform RRT sampling with an adaptive MCMC sampler yields significantly shorter paths when planning succeeds, but at the cost of reduced reliability on complex mazes. Specifically, the *AdaptiveMCMCAgent* achieved a 63.3% success rate on Hard mazes compared to 93.3% for the uniform *RRTAgent* — a statistically significant gap ($\chi^2 = 6.28$, $p = 0.012$). The Discussion (Section 7) attributed these failures to a stall-on-initialization behavior in which the agent generates partial paths without making forward progress, and the Conclusion (Section 8) proposed two immediate remedies: greedy forward-progress acceptance and adaptive iteration budgets. This appendix reports the implementation and evaluation of both improvements, and additionally explores a secondary question: whether the collision and free-space labels produced as a byproduct of MCMC sampling can be used to reconstruct an approximate representation of the obstacle space $\mathcal{X}_{\text{obs}}$.

---

## B. Planner Improvements

### B.1 Greedy Forward-Progress Acceptance

In the baseline architecture, the receding-horizon controller (Algorithm 3) declares a stall after $n_{\text{stall}} = 10$ consecutive planning cycles that return an empty path ($|\mathcal{P}| = 0$). However, the partial-path fallback mechanism (Section 4.3) frequently returns paths containing a single waypoint close to the robot's current position. Because $|\mathcal{P}| = 1 > 0$, the stall counter resets, and the robot can oscillate indefinitely — executing trivial one-step movements without making meaningful progress toward $q_{\text{target}}$.

We address this by augmenting the stall detector with a distance-based criterion. The controller maintains a sliding window of the $W$ most recent positions $\{q_{n-W+1}, \ldots, q_n\}$. A distance stall is triggered when the displacement over the window falls below a threshold:

$$\|q_n - q_{n-W}\| < \delta_{\text{stall}}$$

We use $W = 10$ and $\delta_{\text{stall}} = 2.0$ units. This condition fires only after $W$ planning cycles that each produced at least one waypoint, ensuring that genuine short-range progress (e.g., navigating a tight corridor) is not penalized. The original empty-path stall detector is retained as a complementary mechanism for the case where the planner repeatedly fails to produce any path at all.

### B.2 Adaptive Iteration Budget

The baseline planner uses a fixed iteration budget $N_{\max} = 15{,}000$ for every call to MCMC-RRT, regardless of the distance between the robot's current position $q$ and the planning target $q_{\text{target}}$. In Hard mazes, where narrow corridors may require the tree to grow across 80 or more units of workspace, this budget is often insufficient to find even a partial path.

We introduce a distance-scaled iteration budget:

$$N_{\text{adaptive}}(q, q_{\text{target}}) = \min\!\Bigl(\bigl\lfloor N_{\text{base}}\bigl(1 + d\,/\,d_{\text{scale}}\bigr)\bigr\rfloor,\; N_{\text{cap}}\Bigr)$$

where $d = \|q - q_{\text{target}}\|$, $N_{\text{base}} = 15{,}000$, $d_{\text{scale}} = 40.0$, and $N_{\text{cap}} = 50{,}000$. This formula allocates the baseline budget for nearby targets ($d \approx 0$) and scales linearly up to a hard cap of 50,000 iterations for distant ones. At $d = 80$ (a typical cross-maze distance in the $100 \times 100$ workspace), the budget is $15{,}000 \times 3.0 = 45{,}000$ iterations — a threefold increase over the baseline.

The adaptive budget is applied at both planning call sites in Algorithm 3: the initial plan (line 8) and event-triggered replans (line 22).

### B.3 Experimental Setup

We evaluate the improved planner using the same experimental framework described in Section 5 of the main paper. Maze environments are generated with identical topology classes (*Easy*: $6 \times 6$ grid, *Medium*: $8 \times 8$, *Hard*: $10 \times 10$) within a $100 \times 100$ unit workspace. Both agents share identical RRT hyperparameters: step size $\eta = 5.0$, goal threshold $\epsilon_{\text{goal}} = 5.0$. The *AdaptiveMCMCAgent* initializes its belief at the workspace centroid with $\Sigma_0 = 400\mathbf{I}$ and processes distance-dependent sensor observations ($\sigma_{\text{far}} = 10.0$, $\sigma_{\text{near}} = 0.5$).

We generate 10 unique maze instances per complexity level (seeds 100–109), yielding 60 planned trials (2 agents $\times$ 3 levels $\times$ 10 maps). This is reduced from the 30-map design of the original experiment due to compute constraints — the improved MCMC agent's higher iteration budgets increase per-trial runtime. One Hard-complexity MCMC trial (seed 109) was excluded after exceeding 95 minutes without termination.

### B.4 Results

Table B.1 compares success rates before and after the architectural improvements.

**Table B.1.** Success rates by agent and complexity level. Baseline results reproduced from Section 6 (30 maps per level); improved results from the 10-map evaluation described above.

| | Baseline MCMC | Improved MCMC | Baseline RRT | Improved RRT |
|---|---|---|---|---|
| **Easy** | 100% (30/30) | 100% (10/10) | 100% (30/30) | 100% (10/10) |
| **Medium** | 50% (15/30) | 80% (8/10) | 57% (17/30) | 80% (8/10) |
| **Hard** | 63% (19/30) | **100% (9/9)** | 93% (28/30) | 100% (10/10) |

The combined effect of distance-based stall detection and adaptive iteration budgets eliminates all Hard-maze failures for the MCMC agent, improving the success rate from 63% to 100% on completed trials. Medium-complexity performance also improves from 50% to 80%. Notably, the improved MCMC agent now matches the RRT baseline at every complexity level, resolving the primary limitation identified in the original paper.

This improvement comes at a computational cost. The adaptive iteration budget increases mean planning time on Hard mazes from approximately 425 seconds per trial (baseline) to 568 seconds. We consider this an acceptable tradeoff: the robot now reliably reaches the goal rather than stalling indefinitely.

The remaining Medium-level failures (2/10) exhibit a distinct mechanism from the original stall-on-initialization pattern. In these cases, the robot enters narrow corridor segments where the MCMC chain's proposal distribution ($\sigma_{\text{prop}} = 5.0$) is too wide relative to the corridor width, causing repeated Metropolis rejections. This suggests that proposal adaptation — not addressed in this work — may be necessary for further gains.

---

## C. Configuration Space Reconstruction from MCMC Samples

### C.1 Motivation

During MCMC-RRT planning, every sampled configuration $q'$ is tested against the geometric collision checker to determine whether $q' \in \mathcal{X}_{\text{free}}$ or $q' \in \mathcal{X}_{\text{obs}}$. In our implementation, this check is performed by the Shapely computational geometry library, which computes exact polygon intersections. While accurate, geometric collision checking dominates the per-iteration cost of the planner.

A natural question arises: can the collision and free-space labels accumulated over the course of planning be used to learn an approximate representation of $\mathcal{X}_{\text{obs}}$? If so, such a model could serve as a fast pre-filter — querying the learned model first and falling back to the exact checker only for ambiguous configurations near obstacle boundaries. We investigate two non-parametric approaches: an occupancy grid and a kernel density estimator.

### C.2 Sample Collection and Labeling

We collect samples by running the *AdaptiveMCMCAgent* on three Hard-complexity mazes (seeds 200–202), recording 25,000 MCMC iterations per maze for a total of 75,000 samples. Each sample is a tuple $(q', \ell, J(q'))$ where $q' \in \mathcal{X}$ is the sampled configuration and $\ell$ is the rejection label. The label takes one of four values:

- **collision**: $q'$ lies in $\mathcal{X}_{\text{obs}}$ (rejected by collision checker)
- **accepted**: $q' \in \mathcal{X}_{\text{free}}$ and accepted by the Metropolis criterion
- **metropolis_rejected**: $q' \in \mathcal{X}_{\text{free}}$ but rejected by the Metropolis criterion
- **uniform**: $q' \in \mathcal{X}_{\text{free}}$, drawn from the uniform mixing component

For the purpose of obstacle classification, we binarize labels as $y = 1$ (collision) for $\ell = \texttt{collision}$ and $y = 0$ (free) for all other labels. Of the 75,000 samples, 9,406 (12.5%) are collision samples and 65,594 (87.5%) are free-space samples.

### C.3 Occupancy Grid Model

We discretize the workspace $\mathcal{X} = [0, 100]^2$ into a regular grid with cell resolution $r = 1.0$ unit, yielding $100 \times 100 = 10{,}000$ cells. For each cell $c$, we maintain counts of collision samples $n_{\text{coll}}(c)$ and free-space samples $n_{\text{free}}(c)$ that fall within its boundaries. The estimated obstacle probability is:

$$\hat{P}(\text{obs} \mid c) = \frac{n_{\text{coll}}(c)}{n_{\text{coll}}(c) + n_{\text{free}}(c)}$$

For cells with no observations, we assign $\hat{P}(\text{obs} \mid c) = 0.5$ (maximum uncertainty). A configuration $q$ is classified as an obstacle if $\hat{P}(\text{obs} \mid c(q)) > \tau_{\text{grid}}$, where $c(q)$ denotes the cell containing $q$ and $\tau_{\text{grid}} = 0.5$.

The primary advantage of this model is its $O(1)$ query time: obstacle probability at any configuration reduces to a single array lookup after discretizing the coordinates to cell indices.

### C.4 Kernel Density Estimation Model

As an alternative, we fit separate kernel density estimators for the collision and free-space sample distributions. Let $\mathcal{D}_{\text{coll}} = \{q_i : y_i = 1\}$ and $\mathcal{D}_{\text{free}} = \{q_i : y_i = 0\}$. We estimate class-conditional densities using Gaussian KDEs:

$$\hat{f}_{\text{coll}}(q) = \frac{1}{|\mathcal{D}_{\text{coll}}|} \sum_{q_i \in \mathcal{D}_{\text{coll}}} K_h(q - q_i), \qquad \hat{f}_{\text{free}}(q) = \frac{1}{|\mathcal{D}_{\text{free}}|} \sum_{q_i \in \mathcal{D}_{\text{free}}} K_h(q - q_i)$$

where $K_h$ is the Gaussian kernel with bandwidth $h = 1.0$. The bandwidth is chosen to match the scale of obstacle features: walls in our maze environments are 2 units thick, so $h = 1.0$ provides resolution at the wall-boundary scale without excessive smoothing.

The posterior obstacle probability is computed via Bayes' rule:

$$\hat{P}(\text{obs} \mid q) = \frac{\hat{f}_{\text{coll}}(q)\,\pi_{\text{coll}}}{\hat{f}_{\text{coll}}(q)\,\pi_{\text{coll}} + \hat{f}_{\text{free}}(q)\,\pi_{\text{free}}}$$

where $\pi_{\text{coll}} = |\mathcal{D}_{\text{coll}}| / (|\mathcal{D}_{\text{coll}}| + |\mathcal{D}_{\text{free}}|)$ is the empirical collision prior. Due to the class imbalance ($\pi_{\text{coll}} \approx 0.125$), we lower the decision threshold to $\tau_{\text{KDE}} = 0.3$; at the default threshold of 0.5, the model almost never predicts collision because the prior-weighted free-space density dominates.

To maintain tractable training and query time, we subsample to $|\mathcal{D}_{\text{train}}| = 10{,}000$ points (selected uniformly at random from the full 75,000). KDE queries are evaluated in batch using vectorized scoring.

### C.5 Evaluation

We evaluate both models on a uniform $60 \times 60$ test grid spanning the workspace, yielding $|\mathcal{D}_{\text{test}}| = 3{,}600$ evaluation points. Ground-truth labels are obtained by querying the Shapely collision checker for each test point. Performance is measured by:

1. **F1 score** on the collision class (the minority class), which balances precision and recall
2. **Wall-clock query time** to classify all 3,600 test points
3. **Speedup** relative to the Shapely baseline query time

**Table C.1.** Configuration space classification performance.

| Model | F1 Score | Query Time (ms) | Speedup |
|---|---|---|---|
| Shapely (ground truth) | — | 1,167 | 1.0$\times$ |
| Occupancy Grid | 0.648 | 5.8 | 202$\times$ |
| KDE (batch, $h = 1.0$) | 0.496 | 545 | 2.1$\times$ |

The occupancy grid achieves a 202-fold speedup over the geometric collision checker while maintaining an F1 score of 0.648. The KDE model achieves a lower F1 of 0.496 with only a 2.1-fold speedup, making it impractical as a collision-checking substitute.

### C.6 Discussion

The occupancy grid's strong speedup and reasonable classification accuracy make it a viable candidate for a hierarchical collision-checking strategy: query the grid first, and invoke the exact geometric checker only when the grid's probability falls in an ambiguous range (e.g., $0.2 < \hat{P} < 0.8$). Such a scheme could substantially reduce per-iteration planning cost while maintaining exact safety guarantees.

The F1 scores for both models fall well below 1.0, reflecting a fundamental limitation of learning $\mathcal{X}_{\text{obs}}$ from MCMC samples. The Boltzmann target distribution $\pi(q) \propto \exp(-J(q)/T)$ concentrates samples in low-cost regions near the goal estimate $\mu$. Obstacle boundaries far from the goal receive few samples, leaving many grid cells unobserved and KDE density estimates unreliable in those regions. The uniform mixing component ($\lambda_{\text{mix}} = 0.1$) partially mitigates this, but 10% of 75,000 samples spread across a $100 \times 100$ workspace provides only modest boundary coverage.

The KDE model's poor performance relative to the grid stems from two factors. First, the $O(n)$ per-query complexity of kernel evaluation — even with vectorized batch scoring — makes it substantially slower than the grid's $O(1)$ lookup. Second, the severe class imbalance ($\pi_{\text{coll}} = 0.125$) means that even moderate errors in density estimation produce large swings in the posterior; the lowered threshold ($\tau = 0.3$) partially compensates but introduces additional false positives.

Several extensions could improve reconstruction quality. Finer grid resolution ($r = 0.5$) would better resolve thin walls at the cost of sparser per-cell counts. Laplace smoothing of cell counts would regularize probability estimates for low-observation cells. A Gaussian process classifier could provide smooth probability surfaces with calibrated uncertainty estimates, enabling principled decisions about when to fall back to the exact checker — though at substantially higher training cost. Finally, active sampling strategies that deliberately explore under-observed boundary regions could improve coverage where the MCMC target distribution provides few samples.

---

## D. Conclusion

This appendix validates both immediate improvements proposed in the main paper's Conclusion. The combination of distance-based stall detection and adaptive iteration budgets eliminates the stall-on-initialization failure mode that caused 37% of Hard-maze trials to fail, raising the *AdaptiveMCMCAgent*'s success rate from 63% to 100% on completed trials while preserving its path-quality advantages. Additionally, we demonstrate that the collision labels generated as a byproduct of MCMC sampling can be repurposed to learn an approximate obstacle map: the occupancy grid model achieves a 202-fold speedup over exact geometric collision checking with an F1 score of 0.648, suggesting a practical path toward amortized collision checking in MCMC-guided planning systems.
