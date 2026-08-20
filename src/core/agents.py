"""Concrete implementations of MazeAgent interface for RRT and Adaptive MCMC agents."""

from typing import Optional, List
import numpy as np

from .agent import MazeAgent
from .environment import MazeEnvironment
from ..rrt.planner import RRTPlanner
from ..rrt.navigation import run_closed_loop_navigation
from ..rrt.goal_belief import GoalBelief
from ..rrt.goal_sensor import GoalObservationModel, GoalSensorConfig
from ..samplers.base import BaseSampler
from ..samplers.adaptive_goal_mcmc import AdaptiveGoalMCMCSampler


class RRTAgent(MazeAgent):
    """Wrapper that adapts RRTPlanner to MazeAgent interface.
    
    This class bridges the gap between the evaluation framework's MazeAgent
    interface and the existing RRTPlanner implementation. It extracts
    parameters from MazeEnvironment objects and delegates to RRTPlanner.
    
    Example:
        >>> from src.samplers.uniform import UniformSampler
        >>> from src.env import ContinuousMazeEnvironment
        >>> from src.config import MazeGenerationConfig
        >>> 
        >>> sampler = UniformSampler(seed=42)
        >>> agent = RRTAgent(sampler, step_size=3.0, max_iterations=5000)
        >>> 
        >>> config = MazeGenerationConfig(space_size=(100, 100), grid_size=(10, 10))
        >>> env = ContinuousMazeEnvironment(config)
        >>> 
        >>> path = agent.solve(env)  # Uses MazeAgent interface!
    """
    
    def __init__(
        self,
        sampler: BaseSampler,
        step_size: float = 3.0,
        goal_threshold: float = 3.0,
        max_iterations: int = 10000,
    ):
        """Initialize RRT agent.
        
        Args:
            sampler: Sampler instance (e.g., UniformSampler, AdaptiveGoalMCMCSampler).
            step_size: Maximum step size for tree expansion.
            goal_threshold: Maximum distance to consider goal reached.
            max_iterations: Maximum number of iterations before giving up.
        """
        self.sampler = sampler
        self.step_size = step_size
        self.goal_threshold = goal_threshold
        self.max_iterations = max_iterations
        
        # Internal state for get_next_move()
        self._planner: Optional[RRTPlanner] = None
        self._current_path: Optional[List[np.ndarray]] = None
        self._path_index: int = 0
    
    def solve(self, environment: MazeEnvironment) -> Optional[List[np.ndarray]]:
        """Solve the maze and return a path from start to goal.
        
        Args:
            environment: The maze environment to solve.
        
        Returns:
            List of positions forming a path from start to goal, or None if
            no path found.
        """
        # Extract parameters from environment
        start = environment.get_start()
        goal = environment.get_goal()
        bounds = environment.get_bounds()
        
        # Get obstacles - need to check if environment has get_obstacles()
        if hasattr(environment, 'get_obstacles'):
            obstacles = environment.get_obstacles()
        else:
            raise ValueError(
                f"Environment {type(environment).__name__} must implement "
                "get_obstacles() method for RRTAgent"
            )
        
        # Create planner with extracted parameters
        self._planner = RRTPlanner(
            start=start,
            goal=goal,
            obstacles=obstacles,
            sampler=self.sampler,
            step_size=self.step_size,
            goal_threshold=self.goal_threshold,
            bounds=bounds,
            max_iterations=self.max_iterations,
        )
        
        # Call existing plan() method
        path = self._planner.plan()
        self._current_path = path
        self._path_index = 0
        
        return path
    
    def get_next_move(
        self, current_position: np.ndarray, environment: MazeEnvironment
    ) -> Optional[np.ndarray]:
        """Get the next move from the current position.
        
        This method enables step-by-step execution. If solve() hasn't been
        called yet, it will be called automatically.
        
        Args:
            current_position: Current position in the maze.
            environment: The maze environment being navigated.
        
        Returns:
            Next position to move to, or None if no valid move available or
            goal reached.
        """
        # If no path exists, solve first
        if self._current_path is None:
            self.solve(environment)
        
        if self._current_path is None:
            return None  # No path found
        
        # Check if we've reached the goal
        goal = environment.get_goal()
        if np.linalg.norm(current_position - goal) <= self.goal_threshold:
            return None  # Goal reached
        
        # Find closest waypoint in path to current position
        if self._path_index < len(self._current_path):
            next_waypoint = self._current_path[self._path_index]
            self._path_index += 1
            return next_waypoint
        
        return None  # Path exhausted
    
    def reset(self, seed: Optional[int] = None) -> None:
        """Reset the agent's internal state.
        
        This resets the sampler's random state, which is important for
        stochastic agents that need reproducible behavior across trials.
        
        Args:
            seed: Optional random seed for reproducibility.
        """
        if seed is not None:
            self.sampler.reset(seed)
        self._planner = None
        self._current_path = None
        self._path_index = 0


class AdaptiveMCMCAgent(MazeAgent):
    """Wrapper that adapts run_closed_loop_navigation to MazeAgent interface.
    
    This class bridges the gap between the evaluation framework's MazeAgent
    interface and the existing adaptive MCMC navigation system. It extracts
    parameters from MazeEnvironment objects and delegates to the navigation
    function, then converts the result dictionary to a simple path list.
    
    Example:
        >>> from src.samplers.adaptive_goal_mcmc import AdaptiveGoalMCMCSampler
        >>> from src.rrt.goal_sensor import GoalObservationModel, GoalSensorConfig
        >>> from src.rrt.goal_belief import GoalBelief
        >>> 
        >>> sampler = AdaptiveGoalMCMCSampler(seed=42)
        >>> sensor = GoalObservationModel(GoalSensorConfig(seed=42))
        >>> belief = GoalBelief(mean=np.array([50, 50]), covariance=np.eye(2) * 100)
        >>> 
        >>> agent = AdaptiveMCMCAgent(sampler, sensor, belief)
        >>> path = agent.solve(env)  # Uses MazeAgent interface!
    """
    
    def __init__(
        self,
        sampler: AdaptiveGoalMCMCSampler,
        sensor: GoalObservationModel,
        initial_belief: GoalBelief,
        step_size: float = 3.0,
        goal_threshold: float = 3.0,
        goal_tolerance: float = 5.0,
        max_steps: int = 400,
        execution_horizon: int = 10,
        entropy_threshold: float = 0.1,
        max_iterations: int = 8000,
        min_steps_between_replans: int = 5,
        adaptive_iterations: bool = True,
    ):
        """Initialize Adaptive MCMC agent.
        
        Args:
            sampler: AdaptiveGoalMCMCSampler for planning.
            sensor: GoalObservationModel for generating observations.
            initial_belief: Initial GoalBelief representing uncertainty about goal.
            step_size: Distance robot moves per step during execution.
            goal_threshold: Distance threshold for RRT goal checking.
            goal_tolerance: Distance to true goal to declare success.
            max_steps: Maximum execution steps before giving up.
            execution_horizon: Number of waypoints to execute before replanning.
            entropy_threshold: Entropy reduction triggering early replan.
            max_iterations: Max RRT iterations per planning call.
            min_steps_between_replans: Minimum steps between replans.
        """
        self.sampler = sampler
        self.sensor = sensor
        self.initial_belief = initial_belief
        self.step_size = step_size
        self.goal_threshold = goal_threshold
        self.goal_tolerance = goal_tolerance
        self.max_steps = max_steps
        self.execution_horizon = execution_horizon
        self.entropy_threshold = entropy_threshold
        self.max_iterations = max_iterations
        self.min_steps_between_replans = min_steps_between_replans
        self.adaptive_iterations = adaptive_iterations
        self._last_result: Optional[dict] = None  # Store telemetry from last solve()
    
    def solve(self, environment: MazeEnvironment) -> Optional[List[np.ndarray]]:
        """Solve the maze using adaptive MCMC navigation.
        
        This method runs the full closed-loop navigation system with
        event-triggered replanning based on entropy reduction.
        
        Args:
            environment: The maze environment to solve.
        
        Returns:
            List of positions forming the executed path, or None if
            navigation failed.
        """
        # Extract parameters from environment
        start = environment.get_start()
        goal = environment.get_goal()  # This is the true goal
        bounds = environment.get_bounds()
        
        # Get obstacles
        if hasattr(environment, 'get_obstacles'):
            obstacles = environment.get_obstacles()
        else:
            raise ValueError(
                f"Environment {type(environment).__name__} must implement "
                "get_obstacles() method for AdaptiveMCMCAgent"
            )
        
        # Call existing navigation function
        result = run_closed_loop_navigation(
            sampler=self.sampler,
            sensor=self.sensor,
            true_goal=goal,
            start_pose=start,
            initial_belief=self.initial_belief.copy(),
            obstacles=obstacles,
            bounds=bounds,
            step_size=self.step_size,
            goal_threshold=self.goal_threshold,
            goal_tolerance=self.goal_tolerance,
            max_steps=self.max_steps,
            execution_horizon=self.execution_horizon,
            entropy_threshold=self.entropy_threshold,
            max_iterations=self.max_iterations,
            min_steps_between_replans=self.min_steps_between_replans,
            adaptive_iterations=self.adaptive_iterations,
        )

        # Store full result for telemetry access
        self._last_result = result

        # Extract path from result dictionary
        if result['success']:
            return result['executed_path']
        return None
    
    def get_next_move(
        self, current_position: np.ndarray, environment: MazeEnvironment
    ) -> Optional[np.ndarray]:
        """Get the next move from the current position.
        
        Note: For AdaptiveMCMCAgent, step-by-step execution is complex because
        the navigation system handles replanning internally. This method is
        not fully supported - use solve() for complete navigation.
        
        Args:
            current_position: Current position in the maze.
            environment: The maze environment being navigated.
        
        Returns:
            None (step-by-step not supported for adaptive navigation).
        """
        # Adaptive MCMC navigation is designed to run as a complete loop
        # with internal replanning. Step-by-step execution would require
        # maintaining complex state (belief, sensor observations, etc.)
        # For now, return None to indicate this isn't supported
        return None
    
    def reset(self, seed: Optional[int] = None) -> None:
        """Reset the agent's internal state.

        This resets both the sampler and sensor random states, which is
        critical for reproducible behavior across trials.

        Args:
            seed: Optional random seed for reproducibility.
        """
        if seed is not None:
            self.sampler.reset(seed)
            self.sensor.reset(seed)

    def get_telemetry(self) -> dict:
        """Return rich telemetry from the last navigation run.

        Returns:
            Dictionary containing adaptive navigation telemetry:
            - 'num_replans': Number of replanning events
            - 'num_steps': Total execution steps
            - 'belief_history': List of GoalBelief objects over time
            - 'entropy_history': List of entropy values (trace of covariance)
            - 'final_belief': Final GoalBelief after navigation
            - 'distance_to_goal': Final distance to true goal

            Returns empty dict if solve() hasn't been called yet.
        """
        if self._last_result is None:
            return {}

        return {
            'num_replans': self._last_result['num_replans'],
            'num_steps': self._last_result['num_steps'],
            'belief_history': self._last_result['belief_history'],
            'entropy_history': self._last_result['entropy_history'],
            'final_belief': self._last_result['final_belief'],
            'distance_to_goal': self._last_result['distance_to_goal'],
        }


class MCMCRRTAgent(MazeAgent):
    """Simple MCMC-guided RRT agent for direct comparison with uniform sampling.
    
    This agent uses the AdaptiveGoalMCMCSampler with perfect goal knowledge
    (no belief updates during planning). This provides a fair comparison with
    the uniform sampling baseline - both agents get the same goal information,
    but MCMC uses goal-directed sampling while uniform samples randomly.
    
    For experiments studying goal uncertainty and active perception, use
    AdaptiveMCMCAgent instead.
    """
    
    def __init__(
        self,
        sampler: AdaptiveGoalMCMCSampler,
        step_size: float = 5.0,
        goal_threshold: float = 5.0,
        max_iterations: int = 15000,
        use_goal_belief: bool = True,
    ):
        """Initialize MCMC-guided RRT agent.
        
        Args:
            sampler: AdaptiveGoalMCMCSampler for goal-directed sampling.
            step_size: Maximum step size for tree expansion.
            goal_threshold: Maximum distance to consider goal reached.
            max_iterations: Maximum number of iterations before giving up.
            use_goal_belief: If True, pass goal belief to sampler for MCMC guidance.
                            If False, sampler falls back to center-biased sampling.
        """
        self.sampler = sampler
        self.step_size = step_size
        self.goal_threshold = goal_threshold
        self.max_iterations = max_iterations
        self.use_goal_belief = use_goal_belief
        
        # Internal state
        self._current_path: Optional[List[np.ndarray]] = None
        self._path_index: int = 0
    
    def solve(self, environment: MazeEnvironment) -> Optional[List[np.ndarray]]:
        """Solve the maze using MCMC-guided RRT.
        
        Args:
            environment: The maze environment to solve.
        
        Returns:
            List of positions forming a path from start to goal, or None if
            no path found.
        """
        start = environment.get_start()
        goal = environment.get_goal()
        bounds = environment.get_bounds()
        
        if hasattr(environment, 'get_obstacles'):
            obstacles = environment.get_obstacles()
        else:
            raise ValueError(
                f"Environment {type(environment).__name__} must implement "
                "get_obstacles() method for MCMCRRTAgent"
            )
        
        # Create goal belief centered on the actual goal with low uncertainty
        # This gives MCMC sampler accurate goal information
        goal_belief = None
        sensor_config = None
        
        if self.use_goal_belief:
            goal_belief = GoalBelief(
                mean=goal.copy(),
                covariance=np.eye(2) * 1.0,  # Low uncertainty - we "know" the goal
            )
            sensor_config = GoalSensorConfig(
                far_std=8.0,
                near_std=0.5,
                distance_scale=50.0,
            )
        
        # Create planner with MCMC sampler
        planner = RRTPlanner(
            start=start,
            goal=goal,
            obstacles=obstacles,
            sampler=self.sampler,
            step_size=self.step_size,
            goal_threshold=self.goal_threshold,
            bounds=bounds,
            max_iterations=self.max_iterations,
            goal_belief=goal_belief,
            sensor_config=sensor_config,
        )
        
        path = planner.plan()
        self._current_path = path
        self._path_index = 0
        
        return path
    
    def get_next_move(
        self, current_position: np.ndarray, environment: MazeEnvironment
    ) -> Optional[np.ndarray]:
        """Get the next move from the current position."""
        if self._current_path is None:
            self.solve(environment)
        
        if self._current_path is None:
            return None
        
        goal = environment.get_goal()
        if np.linalg.norm(current_position - goal) <= self.goal_threshold:
            return None
        
        if self._path_index < len(self._current_path):
            next_waypoint = self._current_path[self._path_index]
            self._path_index += 1
            return next_waypoint
        
        return None
    
    def reset(self, seed: Optional[int] = None) -> None:
        """Reset the agent's internal state."""
        if seed is not None:
            self.sampler.reset(seed)
        self._current_path = None
        self._path_index = 0
    
    def get_name(self) -> str:
        """Return agent name."""
        return "MCMCRRTAgent"

