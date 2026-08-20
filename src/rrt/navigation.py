"""Receding horizon navigation with event-triggered replanning.

This module implements closed-loop navigation that integrates perception,
planning, and execution. The system reactively replans when sensor observations
significantly reduce goal uncertainty (event-triggered replanning).
"""

from __future__ import annotations

from typing import List
import numpy as np
from shapely.geometry import Polygon

from .planner import RRTPlanner
from .goal_belief import GoalBelief
from .goal_sensor import GoalObservationModel
from .collision import point_is_free
from ..samplers.base import BaseSampler


def compute_adaptive_iterations(
    current_pose: np.ndarray,
    target: np.ndarray,
    base: int = 15000,
    scale: float = 40.0,
    max_iterations: int = 50000,
) -> int:
    """Scale max_iterations based on distance to target (Solution 2.2).

    Allocates more planning budget when the problem is harder (long distance)
    and less when it is easier (short distance).

    Args:
        current_pose: Current robot position.
        target: Planning target position.
        base: Baseline iteration count (default 15000).
        scale: Distance scale factor (default 40.0).
        max_iterations: Hard cap on iterations (default 50000).

    Returns:
        Adaptive iteration budget capped at max_iterations.
    """
    distance = np.linalg.norm(target - current_pose)
    return min(int(base * (1 + distance / scale)), max_iterations)


def get_valid_target(
    belief: GoalBelief,
    obstacles: List[Polygon],
    bounds: np.ndarray,
    num_samples: int = 100,
) -> np.ndarray:
    """Sample from belief distribution and return valid point closest to mean.

    This function implements robust target selection by sampling multiple points
    from the goal belief distribution and selecting the collision-free point
    that is closest to the belief mean. This prevents planning failures when
    the belief mean falls inside obstacles or unreachable areas.

    Args:
        belief: Current goal belief distribution
        obstacles: List of Shapely Polygon obstacles
        bounds: Workspace limits [[x_min, y_min], [x_max, y_max]]
        num_samples: Number of samples to draw from belief (default: 100)

    Returns:
        Valid target position [x, y] that is collision-free and closest to mean.
        If no valid samples found, returns clamped belief mean.
    """
    # First check if belief mean itself is valid
    clamped_mean = np.clip(belief.mean, bounds[0], bounds[1])
    if point_is_free(clamped_mean, obstacles):
        return clamped_mean

    # Sample points from the belief distribution
    samples = belief.sample(num_samples)

    # Filter for valid (collision-free) samples within bounds
    valid_samples = []
    for sample in samples:
        # Check if sample is within bounds
        if np.all(bounds[0] <= sample) and np.all(sample <= bounds[1]):
            # Check if sample is collision-free
            if point_is_free(sample, obstacles):
                valid_samples.append(sample)

    # If we found valid samples, return the one closest to the mean
    if len(valid_samples) > 0:
        valid_samples_array = np.array(valid_samples)
        distances_to_mean = np.linalg.norm(valid_samples_array - belief.mean, axis=1)
        closest_idx = np.argmin(distances_to_mean)
        return valid_samples_array[closest_idx]

    # Last resort: Try a grid search near the clamped mean
    # Search in a small radius around clamped mean for a valid point
    search_radius = min(np.ptp(bounds, axis=0)) * 0.1  # 10% of smallest dimension
    for _ in range(50):  # Try 50 random points near clamped mean
        offset = np.random.uniform(-search_radius, search_radius, size=belief.dimension)
        candidate = np.clip(clamped_mean + offset, bounds[0], bounds[1])
        if point_is_free(candidate, obstacles):
            return candidate

    # Absolute fallback: return clamped mean even if in collision
    # This will likely cause planning to return partial path
    return clamped_mean


def run_closed_loop_navigation(
    sampler: BaseSampler,
    sensor: GoalObservationModel,
    true_goal: np.ndarray,
    start_pose: np.ndarray,
    initial_belief: GoalBelief,
    obstacles: List[Polygon],
    bounds: np.ndarray,
    step_size: float = 1.0,
    goal_threshold: float = 0.5,
    goal_tolerance: float = 1.0,
    max_steps: int = 100,
    execution_horizon: int = 10,
    entropy_threshold: float = 0.1,
    max_iterations: int = 10000,
    min_steps_between_replans: int = 5,
    adaptive_iterations: bool = False,
) -> dict:
    """Execute receding horizon navigation with event-triggered replanning.

    This function implements a closed perception-planning-execution loop:
    1. Plan path using current goal belief
    2. Execute a short segment (execution_horizon steps)
    3. Collect sensor observations during execution
    4. Update goal belief with new observations
    5. Trigger early replanning if entropy drops significantly
    6. Repeat until goal reached or max_steps exceeded

    Args:
        sampler: Sampler instance (e.g., AdaptiveGoalMCMCSampler) for planning
        sensor: GoalObservationModel for generating noisy goal observations
        true_goal: Ground-truth goal position [x, y] (hidden from planner)
        start_pose: Initial robot position [x, y]
        initial_belief: Initial GoalBelief representing uncertainty about goal
        obstacles: List of Shapely Polygon obstacles
        bounds: Workspace limits [[x_min, y_min], [x_max, y_max]]
        step_size: Distance robot moves per step during execution (default: 1.0)
        goal_threshold: Distance threshold for RRT goal checking (default: 0.5)
        goal_tolerance: Distance to true goal to declare success (default: 1.0)
        max_steps: Maximum execution steps before giving up (default: 100)
        execution_horizon: Number of waypoints to execute before replanning (default: 10)
        entropy_threshold: Entropy reduction triggering early replan (default: 0.1)
        max_iterations: Max RRT iterations per planning call (default: 10000)
        min_steps_between_replans: Minimum steps between replans to prevent thrashing (default: 5)

    Returns:
        dict containing:
            - 'success': bool, whether robot reached goal
            - 'executed_path': List[np.ndarray], actual path robot took
            - 'num_steps': int, total execution steps
            - 'num_replans': int, number of times replanning occurred
            - 'belief_history': List[GoalBelief], belief state at each replan
            - 'entropy_history': List[float], entropy values at each replan
            - 'final_belief': GoalBelief, final goal belief
            - 'distance_to_goal': float, final distance to true goal
    """
    # Initialize state
    current_pose = np.asarray(start_pose, dtype=float).flatten()
    true_goal_array = np.asarray(true_goal, dtype=float).flatten()
    current_belief = initial_belief.copy()
    current_path = None

    # Take multiple initial observations to better localize goal before first planning
    # This significantly improves initial belief accuracy, especially with high uncertainty
    num_initial_observations = 5  # Take 5 observations before planning
    for _ in range(num_initial_observations):
        initial_obs, initial_obs_cov = sensor.observe(current_pose, true_goal_array)
        current_belief.update(initial_obs, initial_obs_cov)

    # Tracking variables
    executed_path = [current_pose.copy()]
    belief_history = [current_belief.copy()]
    entropy_history = [current_belief.compute_entropy()]
    num_replans = 0
    num_steps = 0
    steps_since_last_replan = 0  # Hysteresis counter
    stall_count = 0  # Track consecutive planning cycles that return 0 waypoints
    max_stall_iterations = 10  # Give up after this many empty-path cycles
    # Solution 2.1: distance-based stall tracking
    recent_positions: list = []  # Ring buffer of last N positions
    distance_stall_window = 10  # Number of steps to check for movement
    distance_stall_threshold = 2.0  # Min distance to travel in window

    # Main execution loop
    while num_steps < max_steps:
        # Check if goal reached
        distance_to_true_goal = np.linalg.norm(current_pose - true_goal_array)
        if distance_to_true_goal <= goal_tolerance:
            return {
                'success': True,
                'executed_path': executed_path,
                'num_steps': num_steps,
                'num_replans': num_replans,
                'belief_history': belief_history,
                'entropy_history': entropy_history,
                'final_belief': current_belief,
                'distance_to_goal': distance_to_true_goal,
            }

        # === PLANNING PHASE ===
        # Check if we need to (re)plan
        needs_replan = (
            current_path is None  # Initial planning
            or len(current_path) <= 1  # Exhausted current path
        )

        if needs_replan:
            # Get valid target from belief distribution
            # This prevents planning to targets inside walls or unreachable areas
            target = get_valid_target(current_belief, obstacles, bounds, num_samples=100)

            # Solution 2.2: scale budget with distance when adaptive_iterations=True
            plan_iters = compute_adaptive_iterations(current_pose, target) if adaptive_iterations else max_iterations

            # Plan using valid target
            planner = RRTPlanner(
                start=current_pose,
                goal=target,  # Plan toward valid target (not belief.mean directly)
                obstacles=obstacles,
                sampler=sampler,
                step_size=step_size,
                goal_threshold=goal_threshold,
                bounds=bounds,
                max_iterations=plan_iters,
                goal_belief=current_belief,
                sensor_config=sensor.config,
            )

            current_path = planner.plan(return_partial=True)

            if current_path is None:
                # Planning failed catastrophically (should not happen with return_partial=True)
                print(f"⚠️  CRITICAL: Planning failed completely at step {num_steps}")
                return {
                    'success': False,
                    'executed_path': executed_path,
                    'num_steps': num_steps,
                    'num_replans': num_replans,
                    'belief_history': belief_history,
                    'entropy_history': entropy_history,
                    'final_belief': current_belief,
                    'distance_to_goal': distance_to_true_goal,
                }

            # Check if we got a partial path (didn't reach target)
            if len(current_path) > 0:
                path_end = current_path[-1]
                distance_to_target = np.linalg.norm(path_end - target)
                if distance_to_target > goal_threshold:
                    # Partial path returned (best effort)
                    print(f"ℹ️  Best-effort partial path at step {num_steps}: reached {distance_to_target:.2f} from target")

            num_replans += 1
            steps_since_last_replan = 0  # Reset hysteresis counter

            # Remove current position from path (we're already there)
            if len(current_path) > 0 and np.allclose(current_path[0], current_pose):
                current_path = current_path[1:]

        # === EXECUTION PHASE ===
        # Execute up to execution_horizon waypoints
        waypoints_to_execute = min(execution_horizon, len(current_path))
        entropy_before_execution = current_belief.compute_entropy()

        # Solution 2.1: Check for stall — empty path cycles AND distance stall
        if waypoints_to_execute == 0:
            stall_count += 1
            if stall_count >= max_stall_iterations:
                print(f"⚠️  Navigation stalled after {stall_count} empty-path cycles. Giving up.")
                return {
                    'success': False,
                    'executed_path': executed_path,
                    'num_steps': num_steps,
                    'num_replans': num_replans,
                    'belief_history': belief_history,
                    'entropy_history': entropy_history,
                    'final_belief': current_belief,
                    'distance_to_goal': distance_to_true_goal,
                }
            # Skip execution and try planning again
            continue
        else:
            # Reset empty-path stall counter on ANY progress (even 1 waypoint)
            stall_count = 0

        # Solution 2.1: Distance-based stall detection
        # Track recent positions and check if robot is making physical progress
        recent_positions.append(current_pose.copy())
        if len(recent_positions) > distance_stall_window:
            recent_positions.pop(0)
        if len(recent_positions) == distance_stall_window:
            span = np.linalg.norm(
                np.array(recent_positions[-1]) - np.array(recent_positions[0])
            )
            if span < distance_stall_threshold:
                print(f"⚠️  Distance stall: only moved {span:.2f} units in last {distance_stall_window} steps. Giving up.")
                return {
                    'success': False,
                    'executed_path': executed_path,
                    'num_steps': num_steps,
                    'num_replans': num_replans,
                    'belief_history': belief_history,
                    'entropy_history': entropy_history,
                    'final_belief': current_belief,
                    'distance_to_goal': distance_to_true_goal,
                }

        for i in range(waypoints_to_execute):
            # Move to next waypoint
            target_waypoint = current_path[i]
            current_pose = np.asarray(target_waypoint, dtype=float).flatten()
            executed_path.append(current_pose.copy())
            num_steps += 1
            steps_since_last_replan += 1

            # Collect sensor observation at new position
            observation, obs_covariance = sensor.observe(current_pose, true_goal_array)

            # Update belief with new observation
            current_belief.update(observation, obs_covariance)

            # Check for event-triggered replanning with hysteresis
            entropy_after_observation = current_belief.compute_entropy()
            entropy_reduction = entropy_before_execution - entropy_after_observation

            if entropy_reduction >= entropy_threshold and steps_since_last_replan >= min_steps_between_replans:
                # Significant information gain! Attempt safe replanning
                belief_history.append(current_belief.copy())
                entropy_history.append(entropy_after_observation)

                # Get valid target from updated belief
                replan_target = get_valid_target(current_belief, obstacles, bounds, num_samples=100)

                # Try to replan immediately (Safety Net: don't clear old path yet)
                replan_iters = compute_adaptive_iterations(current_pose, replan_target) if adaptive_iterations else max_iterations
                planner = RRTPlanner(
                    start=current_pose,
                    goal=replan_target,  # Plan toward valid target
                    obstacles=obstacles,
                    sampler=sampler,
                    step_size=step_size,
                    goal_threshold=goal_threshold,
                    bounds=bounds,
                    max_iterations=replan_iters,
                    goal_belief=current_belief,
                    sensor_config=sensor.config,
                )

                new_path = planner.plan(return_partial=True)

                if new_path is not None:
                    # Check if we got a partial path during event-triggered replan
                    if len(new_path) > 0:
                        path_end = new_path[-1]
                        distance_to_target = np.linalg.norm(path_end - replan_target)
                        if distance_to_target > goal_threshold:
                            # Partial path returned (best effort)
                            print(f"ℹ️  Event-triggered replan (step {num_steps}): partial path, reached {distance_to_target:.2f} from target")

                    # Replanning succeeded (or returned partial)! Use new path
                    num_replans += 1
                    steps_since_last_replan = 0

                    # Remove current position from new path (we're already there)
                    if len(new_path) > 0 and np.allclose(new_path[0], current_pose):
                        new_path = new_path[1:]

                    # Replace current path with new path
                    # Keep remaining waypoints from old path if new path is empty
                    if len(new_path) > 0:
                        current_path = new_path
                    # If new path is empty but we have remaining waypoints, keep them
                    # This handles edge case where replanning succeeds but returns minimal path

                    # Break to start executing new path
                    break
                else:
                    # Replanning failed catastrophically - Safety Net: continue with old path
                    print(f"⚠️  Warning: Event-triggered replanning failed completely at step {num_steps}, continuing on current path")
                    # Continue executing remaining waypoints from current_path
                    # Don't break - keep executing the old path

            # Check if goal reached during execution
            if np.linalg.norm(current_pose - true_goal_array) <= goal_tolerance:
                return {
                    'success': True,
                    'executed_path': executed_path,
                    'num_steps': num_steps,
                    'num_replans': num_replans,
                    'belief_history': belief_history,
                    'entropy_history': entropy_history,
                    'final_belief': current_belief,
                    'distance_to_goal': np.linalg.norm(current_pose - true_goal),
                }

            # Check step limit during execution
            if num_steps >= max_steps:
                break

        # Remove executed waypoints from path
        current_path = current_path[waypoints_to_execute:]

        # Record belief state after execution phase
        if len(current_path) == 0:  # Will trigger replanning
            belief_history.append(current_belief.copy())
            entropy_history.append(current_belief.compute_entropy())

    # Exceeded max_steps without reaching goal
    return {
        'success': False,
        'executed_path': executed_path,
        'num_steps': num_steps,
        'num_replans': num_replans,
        'belief_history': belief_history,
        'entropy_history': entropy_history,
        'final_belief': current_belief,
        'distance_to_goal': np.linalg.norm(current_pose - true_goal_array),
    }
