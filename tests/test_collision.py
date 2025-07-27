from shapely.geometry import box 
from src.collision import is_collision

#provide expected behavior for is_collision with assert. If the condition evaluates to False we get an AssertionError exception 
def test_p_inside_boxs():
    obs = [ box(0,4,0,4), box(5, 0, 8, 7) ]
    assert is_collision((5, 5), obs) is True 


def test_p_outside_boxes():
    obs = [box(0,2,0,2)]
    assert is_collision((3, 3), obs) is False