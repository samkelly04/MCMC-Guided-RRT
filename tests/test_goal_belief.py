import numpy as np
import pytest

from src.rrt.goal_belief import GoalBelief


def test_goal_belief_update_moves_mean_toward_observation():
    belief = GoalBelief(
        mean=np.array([0.0, 0.0]),
        covariance=np.eye(2) * 10.0,
    )
    obs = np.array([5.0, 0.0])
    obs_cov = np.eye(2)

    belief.update(obs, obs_cov)

    assert belief.observation_count == 1
    assert belief.mean[0] > 0.0  # shifted toward observation
    assert belief.covariance[0, 0] < 10.0  # uncertainty reduced


def test_goal_belief_expected_distance_increases_with_offset():
    belief = GoalBelief(
        mean=np.array([0.0, 0.0]),
        covariance=np.eye(2),
    )

    close = belief.expected_distance(np.array([0.5, 0.0]))
    far = belief.expected_distance(np.array([5.0, 0.0]))

    assert close < far


def test_goal_belief_expected_distance_minimum_at_mean():
    """Verify that expected distance has its minimum at the belief mean.
    
    The expected_distance method returns pure Euclidean distance ||q - μ||,
    serving as the exploitation term in the composite cost function. This
    ensures the global minimum is at μ (the belief mean).
    
    The previous far-field Taylor expansion had a "volcano effect" where
    the minimum was at a ring around μ, not at μ itself. The new formulation
    delegates exploration to the information gain term, avoiding this issue.
    """
    belief = GoalBelief(
        mean=np.array([0.0, 0.0]),
        covariance=np.eye(2) * 10.0,  # Covariance doesn't affect distance now
    )

    # Cost at the mean should be exactly 0
    cost_at_mean = belief.expected_distance(np.array([0.0, 0.0]))
    
    # Cost at various distances from the mean
    cost_at_1 = belief.expected_distance(np.array([1.0, 0.0]))
    cost_at_3 = belief.expected_distance(np.array([3.0, 0.0]))
    cost_at_5 = belief.expected_distance(np.array([5.0, 0.0]))
    
    # The mean should have the lowest cost (zero)
    assert cost_at_mean == 0.0, "Mean should have zero cost"
    assert cost_at_mean < cost_at_1, "Mean should have lower cost than nearby points"
    assert cost_at_mean < cost_at_3, "Mean should have lower cost than r=3"
    assert cost_at_mean < cost_at_5, "Mean should have lower cost than r=5"
    
    # Cost should be monotonically increasing with distance
    assert cost_at_1 < cost_at_3 < cost_at_5, "Cost should increase monotonically with distance"
    
    # Verify the exact formula: ||q - μ|| (pure Euclidean distance)
    assert np.isclose(cost_at_mean, 0.0), f"Expected 0, got {cost_at_mean}"
    assert np.isclose(cost_at_1, 1.0), f"Expected 1.0, got {cost_at_1}"
    assert np.isclose(cost_at_5, 5.0), f"Expected 5.0, got {cost_at_5}"


def test_goal_belief_expected_distance_is_pure_euclidean():
    """Verify that expected_distance returns pure Euclidean distance.
    
    The expected_distance method now returns ||q - μ|| regardless of
    covariance. Uncertainty-awareness is handled by the information gain
    term in the composite cost function J(q) = ||q - μ|| - λ·I(q, Σ).
    """
    belief = GoalBelief(
        mean=np.array([0.0, 0.0]),
        covariance=np.eye(2) * 100.0,  # Large uncertainty shouldn't affect distance
    )
    
    # Test various distances
    for r in [1.0, 5.0, 10.0, 50.0, 100.0]:
        cost = belief.expected_distance(np.array([r, 0.0]))
        assert np.isclose(cost, r), f"Expected {r}, got {cost}"
    
    # Test 2D distance
    point = np.array([3.0, 4.0])  # Distance = 5.0
    cost = belief.expected_distance(point)
    assert np.isclose(cost, 5.0), f"Expected 5.0, got {cost}"


def test_goal_belief_expected_distance_invariant_to_covariance():
    """Verify that expected_distance doesn't depend on covariance.
    
    Exploration-exploitation tradeoff is managed through the information
    gain term, not through the distance computation.
    """
    mean = np.array([0.0, 0.0])
    point = np.array([5.0, 0.0])
    
    # Create beliefs with different covariances
    belief_low = GoalBelief(mean=mean, covariance=np.eye(2) * 0.01)
    belief_high = GoalBelief(mean=mean, covariance=np.eye(2) * 100.0)
    
    cost_low = belief_low.expected_distance(point)
    cost_high = belief_high.expected_distance(point)
    
    # Both should return the same distance (5.0)
    assert np.isclose(cost_low, 5.0), f"Expected 5.0, got {cost_low}"
    assert np.isclose(cost_high, 5.0), f"Expected 5.0, got {cost_high}"
    assert np.isclose(cost_low, cost_high), "Distance should not depend on covariance"


def test_goal_belief_dimension_mismatch_raises():
    belief = GoalBelief(
        mean=np.array([0.0, 0.0]),
        covariance=np.eye(2),
    )

    with pytest.raises(ValueError):
        belief.update(np.array([1.0, 2.0, 3.0]), np.eye(3))

