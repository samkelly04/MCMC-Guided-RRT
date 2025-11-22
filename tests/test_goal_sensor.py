import numpy as np

from src.rrt.goal_sensor import GoalObservationModel, GoalSensorConfig


def test_noise_decreases_with_distance():
    model = GoalObservationModel(
        GoalSensorConfig(far_std=5.0, near_std=0.5, distance_scale=10.0, seed=1)
    )

    robot_far = np.array([-10.0, 0.0])
    robot_near = np.array([9.0, 0.0])
    goal = np.array([10.0, 0.0])

    _, cov_far = model.observe(robot_far, goal)
    _, cov_near = model.observe(robot_near, goal)

    assert cov_far[0, 0] > cov_near[0, 0]


def test_observation_dimensions_match_inputs():
    model = GoalObservationModel(GoalSensorConfig(seed=42))
    robot = np.array([0.0, 0.0])
    goal = np.array([5.0, -3.0])

    observation, covariance = model.observe(robot, goal)

    assert observation.shape == goal.shape
    assert covariance.shape == (2, 2)


def test_reset_reproduces_observations():
    config = GoalSensorConfig(seed=7)
    model = GoalObservationModel(config)
    robot = np.array([0.0, 0.0])
    goal = np.array([1.0, 1.0])

    obs1, _ = model.observe(robot, goal)
    model.reset()
    obs2, _ = model.observe(robot, goal)

    np.testing.assert_allclose(obs1, obs2)

