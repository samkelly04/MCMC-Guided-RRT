import numpy as np
import pytest
from shapely.geometry import box

from src.rrt.planner import RRTPlanner
from src.samplers.base import BaseSampler


class GoalDirectedSampler(BaseSampler):
    """Deterministic sampler that always proposes the goal position."""

    def __init__(self, goal: np.ndarray):
        self.goal = np.array(goal, dtype=float)

    def sample(self, bounds: np.ndarray) -> np.ndarray:
        return self.goal

    def reset(self, seed: int) -> None:  # pragma: no cover - deterministic sampler
        pass


def test_rrt_planner_finds_simple_path():
    sampler = GoalDirectedSampler(goal=np.array([5.0, 0.0]))
    planner = RRTPlanner(
        start=np.array([0.0, 0.0]),
        goal=np.array([5.0, 0.0]),
        obstacles=[],
        sampler=sampler,
        step_size=1.0,
        goal_threshold=0.5,
        bounds=np.array([[0.0, -1.0], [6.0, 1.0]]),
        max_iterations=50,
    )

    path = planner.plan()

    assert path is not None
    assert np.allclose(path[0], np.array([0.0, 0.0]))
    assert np.allclose(path[-1], np.array([5.0, 0.0]))
    assert len(path) >= 2


def test_rrt_planner_raises_when_start_in_collision():
    obstacles = [box(-0.5, -0.5, 0.5, 0.5)]
    sampler = GoalDirectedSampler(goal=np.array([5.0, 0.0]))

    with pytest.raises(ValueError, match="Start position is in collision"):
        RRTPlanner(
            start=np.array([0.0, 0.0]),
            goal=np.array([5.0, 0.0]),
            obstacles=obstacles,
            sampler=sampler,
            step_size=1.0,
            goal_threshold=0.5,
            bounds=np.array([[-1.0, -1.0], [6.0, 1.0]]),
            max_iterations=50,
        )


def test_rrt_planner_returns_none_when_path_blocked():
    obstacles = [box(0.4, -1.0, 0.6, 1.0)]
    sampler = GoalDirectedSampler(goal=np.array([5.0, 0.0]))
    planner = RRTPlanner(
        start=np.array([0.0, 0.0]),
        goal=np.array([5.0, 0.0]),
        obstacles=obstacles,
        sampler=sampler,
        step_size=0.5,
        goal_threshold=0.25,
        bounds=np.array([[-1.0, -1.0], [6.0, 1.0]]),
        max_iterations=20,
    )

    assert planner.plan() is None

