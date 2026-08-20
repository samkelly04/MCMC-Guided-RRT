# MCMC Theory for Goal-Directed RRT Planning

This document explains the Markov Chain Monte Carlo (MCMC) theory underlying our adaptive goal-directed sampler.

## The Problem: Sampling from a Complex Distribution

In RRT planning, we want to sample points that are:
- **Near the goal** (to guide exploration toward the target)
- **Away from obstacles** (to avoid collisions)
- **Adaptive** (as our belief about the goal location improves)

We can't just sample uniformly - we want a **biased distribution** that favors "good" regions, while ensuring our tree remains connected.

## The Target Distribution: Boltzmann Distribution

We define our target distribution using a **Boltzmann distribution**:

```
π(q) ∝ exp(-J(q) / T)
```

Where:
- **q** is a point in configuration space (e.g., [x, y])
- **J(q)** is a cost function 
- **T** is a temperature parameter

### Cost Function Components

Our cost function combines:

1. **Goal proximity**: `J_goal(q) = E[distance(q, q_goal) | observations]`
   - Uses expected distance under goal belief
   - Lower when closer to expected goal location

2. **Obstacle clearance**: `J_clearance(q) = 1 / max(clearance(q), ε)`
   - Encourages staying away from obstacles
   - Higher clearance → lower cost

3. **Hard constraints**: `J(q) = +∞` if out-of-bounds or colliding
   - These states get zero probability

Total cost: `J(q) = λ_goal × J_goal(q) + λ_clearance × J_clearance(q)`

### Temperature Parameter

The temperature **T** controls the distribution's "sharpness":
- **High T**: Flatter distribution → cost differences matter less → more exploration
- **Low T**: Sharper distribution → cost differences matter more → more exploitation

Think of it like annealing: start with high T (explore broadly), then decrease T (focus on promising regions).

## Why We Need MCMC

The problem: We can't directly sample from π(q) because:
1. It's not a standard distribution (like Gaussian or uniform)
2. We only know it up to a normalization constant (the `∝` symbol)
3. Computing the normalization constant would require integrating over the entire space

**Solution**: Use MCMC to build a Markov chain that converges to π(q).

## Markov Chain Monte Carlo (MCMC) Basics

### What is a Markov Chain?

A **Markov chain** is a sequence of states where each state depends only on the previous state:

```
q₀ → q₁ → q₂ → q₃ → ...
```

The key property: `P(q_{t+1} | q_t, q_{t-1}, ..., q₀) = P(q_{t+1} | q_t)`

Each transition is "memoryless" - only the current state matters.

### The Goal: Stationary Distribution

After many steps, a well-designed Markov chain will converge to a **stationary distribution** π(q). This means:
- If we start from π(q), we stay in π(q)
- If we start from anywhere, we eventually converge to π(q)

**Key insight**: If we can design a Markov chain whose stationary distribution is our target π(q), then samples from the chain will eventually come from π(q)!

## Metropolis Algorithm

The Metropolis algorithm is a way to construct such a Markov chain.

### Algorithm Steps

1. **Start** with an initial state q₀
2. **Propose** a new state q' from a proposal distribution Q(q' | q_current)
3. **Compute acceptance probability**: α = min(1, π(q') / π(q_current))
4. **Accept or reject**:
   - Accept with probability α → move to q'
   - Reject with probability (1 - α) → stay at q_current
5. **Repeat** from step 2

### Why This Works

The Metropolis rule ensures **detailed balance**:

```
π(q) × P(q' | q) = π(q') × P(q | q')
```

This means the chain is **reversible** - it's equally likely to go from q to q' as from q' to q (when weighted by probabilities).

Detailed balance guarantees that π(q) is the stationary distribution.

### Proposal Distribution

We use a **symmetric Gaussian proposal**:

```
q' = q_current + N(0, σ²)
```

Where σ (proposal_std) controls how far we "jump" each step:
- **Large σ**: Big jumps → more exploration, but lower acceptance rate
- **Small σ**: Small jumps → higher acceptance rate, but slower exploration

The symmetry (Q(q'|q) = Q(q|q')) is important - it makes the acceptance rule simple.

### Acceptance Rule Explained

The acceptance probability `α = min(1, π(q') / π(q_current))` means:

- **If π(q') > π(q_current)**: α = 1 → always accept (proposal is better)
- **If π(q') < π(q_current)**: α < 1 → accept with probability α (sometimes accept worse proposals)

**Why accept worse proposals?** To avoid getting stuck in local minima! If we only accepted better proposals, we'd get trapped in the first good region we find.

## Metropolis-within-Gibbs

Our implementation uses **Metropolis-within-Gibbs**, which means:
- We update all dimensions at once (not one at a time like pure Gibbs)
- But we use Metropolis acceptance for the update
- The name comes from combining Gibbs-style updates with Metropolis acceptance

In practice, this is just the standard Metropolis algorithm applied to our multi-dimensional space.

## Convergence and Burn-in

### Burn-in Period

Initially, the chain starts from a random point and needs time to "find" the high-probability regions. This initial period is called **burn-in**.

After burn-in, samples come from (approximately) the target distribution.

### Mixing Time

The **mixing time** is how long it takes for the chain to "forget" its starting point and converge to the stationary distribution.

Factors affecting mixing time:
- **Proposal size**: Too small → slow mixing, too large → low acceptance
- **Temperature**: Higher T → faster mixing (flatter distribution)
- **Cost landscape**: Rugged landscapes → slower mixing

### Practical Considerations

In RRT planning:
- We don't need perfect convergence (we're not doing statistical inference)
- We just need samples that are "good enough" (biased toward goal, away from obstacles)
- The chain state persists across RRT iterations, so we get "warm starts"

## Expected Distance Under Goal Belief

A key component of our cost function is the **expected distance**:

```
E[distance(q, q_goal) | observations] = ∫ distance(q, q_goal) × P(q_goal | observations) dq_goal
```

### Why Expected Distance?

We don't know the true goal location - we only have a **belief distribution** P(q_goal | observations) = N(μ, Σ).

The expected distance averages over all possible goal locations, weighted by probability:
- If goal is likely at location A → contributes more to expectation
- If goal is unlikely at location B → contributes less

This is **risk-neutral** decision making: we optimize for average performance.

### Computation

For goal-directed planning under uncertainty, we use a composite cost function:

```
J(q) = ||q - μ|| - λ_info × I(q, Σ)
```

Where:
- `||q - μ||` is the pure distance to the belief mean (exploitation term)
- `I(q, Σ)` is the information gain at position q (exploration term)
- `λ_info` controls the exploration-exploitation balance

This formulation has important properties:
- **Minimum at μ**: The exploitation term has its minimum at the belief mean
- **Explicit control**: The λ_info parameter directly controls exploration vs exploitation
- **Aligned objectives**: Both terms favor the goal region (distance explicitly, info gain implicitly through sensor model)
- **Adaptive**: As Σ decreases, information gain diminishes, converging to pure distance

**Note on alternative approximations**: The far-field Taylor expansion `||q - μ|| + trace(Σ)/(2||q - μ||)` 
creates a "volcano effect" where the minimum is at a ring around μ rather than at μ itself. This 
conflicts with information gain (which is maximized near μ) and creates uncontrollable repulsive 
behavior. The explicit separation of exploitation (distance) and exploration (info gain) avoids this issue.

## Summary

1. **Target distribution**: Boltzmann π(q) ∝ exp(-J(q) / T) encodes what we want
2. **MCMC approach**: Build a Markov chain that converges to π(q)
3. **Metropolis algorithm**: Propose new states, accept/reject based on probability ratio
4. **Chain state**: Maintains "memory" across samples for efficient exploration
5. **Adaptive**: Cost function uses goal belief that updates with observations

The result: Samples are more likely to be near the goal and away from obstacles, while still allowing exploration of the space.

