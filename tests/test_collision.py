# tests/test_collision.py

import numpy as np
from shapely.geometry import box
from src.collision import (
    is_collision,
    point_is_free,
    segment_is_collision,
    segment_is_free,
)

# -------------------------
# POINT COLLISION (uses .contains → boundary is FREE)
# -------------------------

def test_point_inside_is_collision():
    obs = [box(0, 0, 4, 4)]
    assert is_collision((2, 2), obs) is True
    assert point_is_free((2, 2), obs) is False

def test_point_outside_is_not_collision():
    obs = [box(0, 0, 2, 2)]
    assert is_collision((3, 3), obs) is False
    assert point_is_free((3, 3), obs) is True

def test_point_on_edge_is_not_collision_with_contains():
    # Because is_collision uses Polygon.contains, boundary points are NOT collisions.
    obs = [box(0, 0, 4, 4)]
    assert is_collision((0, 2), obs) is False
    assert point_is_free((0, 2), obs) is True

def test_point_on_corner_is_not_collision_with_contains():
    obs = [box(0, 0, 4, 4)]
    assert is_collision((0, 0), obs) is False
    assert point_is_free((0, 0), obs) is True

def test_point_multiple_obstacles_any_inside_triggers_collision():
    obs = [box(0, 0, 4, 4), box(5, 0, 8, 7)]
    assert is_collision((6, 1), obs) is True
    assert point_is_free((6, 1), obs) is False

def test_numpy_point_supported():
    obs = [box(0, 0, 4, 4)]
    p_out = np.array([5.0, 5.0])
    p_in  = np.array([1.0, 1.0])
    assert is_collision(p_out, obs) is False
    assert is_collision(p_in,  obs) is True

# -------------------------
# SEGMENT COLLISION (uses .intersects → touch/cross = collision)
# -------------------------

def test_segment_crosses_box_is_collision():
    obs = [box(10, 10, 20, 20)]
    a, b = (0, 15), (30, 15)
    assert segment_is_collision(a, b, obs) is True
    assert segment_is_free(a, b, obs) is False

def test_segment_clear_gap_is_free():
    obs = [box(10, 10, 20, 20)]
    a, b = (0, 0), (9, 9)
    assert segment_is_collision(a, b, obs) is False
    assert segment_is_free(a, b, obs) is True

def test_segment_touches_corner_is_collision():
    obs = [box(10, 10, 20, 20)]
    a, b = (0, 0), (10, 10)  # touches at corner → intersects == True
    assert segment_is_collision(a, b, obs) is True
    assert segment_is_free(a, b, obs) is False

def test_segment_touches_edge_is_collision():
    obs = [box(10, 10, 20, 20)]
    a, b = (0, 10), (12, 10)  # along the bottom edge → intersects == True
    assert segment_is_collision(a, b, obs) is True
    assert segment_is_free(a, b, obs) is False

def test_degenerate_segment_outside_is_free():
    obs = [box(10, 10, 20, 20)]
    a = b = (5, 5)
    assert segment_is_collision(a, b, obs) is False
    assert segment_is_free(a, b, obs) is True

def test_degenerate_segment_inside_is_collision():
    obs = [box(10, 10, 20, 20)]
    a = b = (12, 12)
    assert segment_is_collision(a, b, obs) is True
    assert segment_is_free(a, b, obs) is False

def test_segment_multiple_obstacles_any_intersection_is_collision():
    obs = [box(0, 0, 4, 4), box(6, 0, 10, 4)]
    a, b = (2, 2), (8, 2)  # intersects both boxes
    assert segment_is_collision(a, b, obs) is True
    assert segment_is_free(a, b, obs) is False