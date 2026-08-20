Collision Checker Function:
  Inputs: 
    1. A single point in 2D space
    2. List of Shapely Polygon objects representing walls/obstacles from the maze generator
  Returns: Boolean telling us whether or not the point is within any of the obstacles. 
    1. Returns TRUE if the point lies inside any of the obstacles 
    2. Returns FALSE if the point is not within any of the obstacles 
  Functionality: When running RRT we must know whether or not our randomly generated points lie in obstacles. 
  When running RRT the planning loop will call this function N times if N is the number of points we generate to build our tree.
  Uses Shapely geometry operations: Point.contains() for point collision, Polygon.intersects() for segment collision.

  Unit Testing: Comprehensive tests for point and segment collision detection with boundary cases.
  
Environment Generator:
  Generates maze-like environments using grid-based random walk algorithm.
  Returns: List of Shapely Polygon objects representing walls in continuous space.
  Inputs:
    - space_size: (width, height) of workspace
    - grid_size: (rows, cols) number of cells in grid
    - wall_thickness: Width of wall segments
    - seed: Random seed for reproducibility
  Features:
    - Guaranteed connectivity (at least one path exists)
    - Configurable dimensions
    - Continuous space (RRT samples from full space, not just grid cells) 

Hybrid Adaptive Goal MCMC Sampler:
  System Overview:
    The Hybrid Adaptive Goal MCMC Sampler addresses a critical theoretical flaw in pure MCMC-based 
    sampling: the "Local Minima Trap." When a sampler strictly follows the cost gradient encoded 
    in the Boltzmann distribution, it can become trapped in local minima of the cost landscape. 
    For example, in environments with U-shaped obstacles or complex maze structures, the MCMC chain 
    may converge to a region that appears optimal locally but prevents the tree from exploring 
    alternative paths that lead to the goal. This behavior violates probabilistic completeness, 
    a fundamental property required for RRT algorithms that guarantees a solution will be found 
    if one exists (given infinite time).
    
    The solution implements Lambda Biasing, a hybrid sampling strategy that maintains the 
    efficiency benefits of goal-directed MCMC sampling while preserving probabilistic completeness. 
    At each sampling step, the algorithm uses a mixing parameter λ (uniform_mixing_rate) to 
    probabilistically choose between two sampling modes: Global Uniform Sampling for exploration 
    and Local MCMC Sampling for exploitation. This ensures that there is always a non-zero 
    probability of sampling any point in the free configuration space, allowing the RRT tree to 
    escape local traps and maintain connectivity guarantees. The adaptive nature of the sampler 
    comes from its integration with the Goal Belief system, which updates the target distribution 
    as new sensor observations refine the estimate of the goal location.

  Mathematical Details:
    
    Target Distribution (Boltzmann):
      The sampler targets a Boltzmann distribution π(q) ∝ exp(-J(q) / T), where J(q) is a cost 
      function combining goal proximity, obstacle clearance, and information gain, and T is a 
      temperature parameter controlling exploration-exploitation tradeoff. The cost function 
      J(q) = λ_goal × J_goal(q) + λ_clearance × J_clearance(q) - λ_info × I(q), where J_goal(q) 
      uses the expected distance under the goal belief distribution, J_clearance(q) penalizes 
      proximity to obstacles, and I(q) is the information gain (entropy reduction) from observing 
      the goal at query point q. The information gain term rewards sampling locations that would 
      maximally reduce goal belief uncertainty, enabling active perception strategies. Hard 
      constraints assign infinite cost to out-of-bounds or colliding configurations.
    
    Hybrid Sampling Strategy (Lambda Biasing):
      The core innovation is the mixture model that combines MCMC sampling with uniform sampling:
      
        x_new ~ (1 - λ) · MCMC(π) + λ · Uniform(𝒳_free)
      
      where λ ∈ [0, 1] is the uniform_mixing_rate parameter, MCMC(π) denotes sampling from the 
      Boltzmann distribution π(q) using Metropolis-within-Gibbs, and Uniform(𝒳_free) denotes 
      uniform sampling over the collision-free configuration space.
      
      This mixture model restores Probabilistic Completeness by ensuring that for any point q in 
      the free space 𝒳_free, there exists a non-zero probability of sampling it. Specifically, 
      P(sample = q) ≥ λ · (1 / |𝒳_free|) > 0 for all q ∈ 𝒳_free, where |𝒳_free| is the measure 
      of the free space. This guarantee holds regardless of the cost landscape structure, 
      preventing the sampler from becoming permanently trapped in local minima.
      
      The mixing parameter λ provides a tunable tradeoff between exploration and exploitation:
      - High λ (e.g., λ = 0.3): More uniform exploration, better completeness guarantees, 
        but slower convergence to goal-directed paths.
      - Low λ (e.g., λ = 0.05): More MCMC exploitation, faster goal-directed behavior, 
        but requires careful tuning to avoid local traps.
      - Adaptive λ: Future work may explore adapting λ based on planning progress or 
        environment complexity.
    
    Metropolis-within-Gibbs Algorithm:
      When the sampler selects MCMC mode (with probability 1 - λ), it performs a Metropolis step:
      1. Propose a new state q' = q_current + N(0, σ²I) using symmetric Gaussian proposal.
      2. Compute log acceptance probability: log α = min(0, log π(q') - log π(q_current)).
      3. Accept q' with probability min(1, exp(log α)), otherwise remain at q_current.
      
      The chain state persists across RRT iterations, providing warm-start benefits that reduce 
      burn-in time. The proposal standard deviation σ (proposal_std) controls the step size of 
      the Markov chain, balancing exploration within the cost landscape against acceptance rates.
    
    Integration with Goal Belief System:
      The cost function J_goal(q) uses the GoalBelief.expected_distance(q) method, which 
      computes E[||q - q_goal|| | observations] under the current Gaussian belief N(μ, Σ). 
      The information gain term I(q) uses GoalBelief.predict_information_gain(q, sensor_config) 
      to compute hypothetical covariance reduction, enabling the sampler to bias toward regions 
      that maximize expected information gain about the goal location. As the Goal Belief system 
      receives new sensor observations and performs Kalman-style updates, the mean μ converges 
      toward the true goal and the covariance Σ decreases. This causes the expected distance 
      function to become more focused, which in turn makes the Boltzmann distribution π(q) more 
      concentrated around promising regions. The hybrid sampler adapts automatically to these 
      changes while maintaining its completeness guarantees through the uniform mixing component.

  Algorithm Logic:
    
    Sample Method Implementation:
      The sample() method implements the hybrid sampling strategy through the following logic flow:
      
      1. Generate a uniform random number r ~ Uniform(0, 1).
      
      2. If r < λ (uniform_mixing_rate):
         a. Generate a uniform random sample q_uniform ~ Uniform(bounds).
         b. If obstacles are provided, ensure q_uniform is collision-free (retry if necessary).
         c. Return q_uniform.
         d. Reset MCMC chain state to None (to prevent stale chain state).
      
      3. Else (r ≥ λ):
         a. If MCMC chain state is None:
            - Initialize chain state q_current by sampling uniformly from collision-free space.
            - Return q_current.
         b. Else:
            - Propose new state: q_proposed = q_current + N(0, σ²I), clamped to bounds.
            - Compute log acceptance: log α = log π(q_proposed) - log π(q_current).
            - Accept q_proposed with probability min(1, exp(log α)).
            - Update chain state: q_current = q_proposed (if accepted) or remain unchanged.
            - Return q_current.
      
      The state dictionary passed to sample() contains:
      - 'goal_belief': GoalBelief object with current mean and covariance.
      - 'obstacles': List of Shapely Polygon obstacles for collision checking and clearance computation.
      - 'sensor_config': GoalSensorConfig object or dict with sensor parameters (far_std, near_std, 
        distance_scale) for information gain prediction.
      
      This implementation ensures that every call to sample() has probability λ of generating 
      a uniform exploration sample, regardless of the current MCMC chain state or cost landscape. 
      The uniform samples provide "escape routes" from local minima, while the MCMC samples 
      provide efficient goal-directed exploration when the chain is in promising regions.

  Safety and Completeness Properties:
    
    Probabilistic Completeness:
      The hybrid sampler guarantees probabilistic completeness for the RRT algorithm. For any 
      two points q_start and q_goal in the free space 𝒳_free, if a collision-free path exists, 
      the probability that RRT finds it approaches 1 as the number of iterations approaches 
      infinity. This property holds because the uniform mixing component ensures that every 
      point in 𝒳_free has non-zero sampling probability, allowing the tree to eventually 
      explore all reachable regions.
    
    Local Minima Escape:
      The uniform mixing rate λ provides a mechanism to escape local minima. When the MCMC chain 
      becomes trapped in a local cost minimum (e.g., inside a U-shaped obstacle), uniform 
      samples can "jump" the chain to distant regions, breaking the local correlation structure. 
      This escape mechanism is probabilistic but guaranteed to occur with probability λ at each 
      step, ensuring that the tree can eventually explore alternative paths.
    
    Adaptive Behavior:
      The sampler maintains adaptive behavior through its integration with the Goal Belief system. 
      As sensor observations refine the goal estimate, the Boltzmann distribution becomes more 
      focused, and the MCMC component naturally guides exploration toward the goal. However, 
      the uniform mixing component ensures that this adaptation does not compromise completeness, 
      as exploration of the entire free space remains possible at all times.

Goal Belief System:
  System Overview:
    The Goal Belief System maintains a probabilistic representation of the goal location using 
    a Gaussian belief distribution N(μ, Σ), where μ is the mean (best estimate) and Σ is the 
    covariance matrix (uncertainty). The system transitions from a "Passive" estimation system 
    to an "Information-Aware" (Active Perception) system by incorporating entropy reduction 
    (information gain) predictions that enable the planner to actively choose sampling locations 
    that maximize expected information gain about the goal location.
    
    The system integrates with the Hybrid Adaptive Goal MCMC Sampler through two mechanisms:
    1. **Expected Distance**: Provides goal proximity cost term J_goal(q) using expected distance 
       under current belief, enabling goal-directed exploration.
    2. **Information Gain Prediction**: Provides entropy reduction term I(q) that rewards 
       sampling locations where goal observations would maximally reduce uncertainty, enabling 
       active perception strategies.

  Mathematical Details:
    
    Kalman-Style Belief Updates:
      The system maintains a Gaussian belief P(q_goal | observations_{1:t}) = N(μ_t, Σ_t) that 
      updates sequentially as new sensor observations arrive. Each update follows a Kalman filter:
      
      1. Innovation: ν_t = z_t - μ_{t-1} (difference between observation and prior mean)
      2. Innovation Covariance: S_t = Σ_{t-1} + Σ_obs (total uncertainty in innovation)
      3. Kalman Gain: K_t = Σ_{t-1} · S_t^{-1} (how much to trust the observation)
      4. Mean Update: μ_t = μ_{t-1} + K_t · ν_t (shift toward observation)
      5. Covariance Update: Σ_t = (I - K_t) · Σ_{t-1} (reduce uncertainty)
      
      The covariance update is deterministic and depends only on uncertainty levels (prior 
      covariance and observation noise), not on the actual measurement values. This ensures 
      predictable uncertainty reduction regardless of what value is observed.
    
    Expected Distance Computation:
      The expected_distance(q) method returns the pure Euclidean distance ||q - μ|| to the 
      belief mean. This serves as the exploitation term in the composite cost function:
      
        J(q) = ||q - μ|| - λ_info · I(q, Σ)
      
      This formulation provides explicit separation of concerns:
      - Exploitation (distance term): Convex function with minimum at μ
      - Exploration (information gain term): Rewards uncertainty reduction, controlled by λ_info
      
      As observations accumulate and Σ decreases, achievable information gain diminishes, 
      causing the cost function to converge naturally to pure distance-to-goal.
    
    Information Gain Prediction (Active Perception):
      The predict_information_gain(query_point, sensor_config) method computes how much the belief 
      uncertainty would decrease if the robot were to observe the goal from query_point. This is 
      a hypothetical calculation that does not modify the actual belief state.
      
      Algorithm:
      1. Compute distance: dist = ||query_point - self.mean||
      2. Get hypothetical observation noise: σ = sensor_config.near_std + 
         (sensor_config.far_std - sensor_config.near_std) · min(dist / sensor_config.distance_scale, 1.0)
      3. Observation covariance: R = σ² · I (isotropic)
      4. Innovation covariance: S = Σ_curr + R
      5. Kalman Gain: K = Σ_curr · S^{-1}
      6. Hypothetical posterior covariance: Σ_new = (I - K) · Σ_curr
      7. Information gain: I(q) = trace(Σ_curr) - trace(Σ_new)
      
      The information gain represents the expected reduction in uncertainty (measured by trace 
      of covariance) from taking a measurement at the query point. Higher information gain 
      indicates that observations from that location would provide more valuable information about 
      the goal location.

  Integration with MCMC Sampler:
    
    Cost Function Integration:
      The Goal Belief System provides two terms to the MCMC sampler's cost function:
      
      - **Goal Proximity Term**: J_goal(q) = expected_distance(q)
        * Encourages sampling near the expected goal location
        * Adapts as belief mean converges toward true goal
        
      - **Information Gain Term**: I(q) = predict_information_gain(q, sensor_config)
        * Rewards sampling at locations that maximize uncertainty reduction
        * Enables active perception: planner actively seeks informative measurements
        * Subtracted from cost (negative term) since lower cost is better
        
      Combined cost: J(q) = λ_goal · J_goal(q) + λ_clearance · J_clearance(q) - λ_info · I(q)
      
      The λ_info parameter controls the tradeoff between goal-directed exploration (high λ_goal) 
      and information-seeking behavior (high λ_info). When λ_info is large, the sampler prioritizes 
      locations that reduce goal uncertainty, even if they are not directly toward the expected 
      goal location.
    
    Active Perception Behavior:
      The information gain term transforms the planner from passive goal-seeking to active 
      information-seeking. Instead of only asking "where do I think the goal is?", the planner 
      also asks "where should I look to learn the most about where the goal is?". This is 
      particularly valuable when:
      
      - Initial goal estimates are highly uncertain (large covariance)
      - Sensor accuracy improves with proximity (distance-dependent noise)
      - Multiple observation opportunities exist before committing to a path
      
      The planner naturally balances exploitation (moving toward expected goal) with exploration 
      (seeking informative measurements), leading to more efficient convergence as uncertainty 
      reduces over time.

  Sensor Model:
    
    Distance-Dependent Noise:
      The GoalObservationModel generates noisy observations with distance-dependent noise:
      
        σ(distance) = near_std + (far_std - near_std) · min(distance / distance_scale, 1.0)
        
      This models sensors that become more accurate as the robot approaches the goal. The 
      observation covariance R = σ² · I is isotropic (independent noise in each dimension).
      
      The distance-dependent noise model ensures that:
      - Far measurements have high uncertainty (σ ≈ far_std)
      - Near measurements have low uncertainty (σ ≈ near_std)
      - Information gain predictions accurately reflect sensor capabilities
      
      This creates a natural incentive for the planner to approach the goal: not only does 
      proximity reduce expected distance, but it also enables more accurate measurements that 
      further reduce uncertainty.


