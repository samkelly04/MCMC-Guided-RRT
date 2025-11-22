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


def test_goal_belief_dimension_mismatch_raises():
    belief = GoalBelief(
        mean=np.array([0.0, 0.0]),
        covariance=np.eye(2),
    )

    with pytest.raises(ValueError):
        belief.update(np.array([1.0, 2.0, 3.0]), np.eye(3))

