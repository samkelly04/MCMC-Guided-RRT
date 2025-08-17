from shapely.geometry import box 
from src import collision


#provide expected behavior for is_collision with assert. If the condition evaluates to False we get an AssertionError exception 
def test_p_inside_boxs():
    obs = [ box(0,0,4,4), box(5, 0, 8, 7) ]
    assert collision.is_collision((5, 5), obs) is True 


def test_p_outside_boxes():
    obs = [box(0,0,2,2)]
    assert collision.is_collision((3, 3), obs) is False


def test_p_is_free():
    obs = [box(0,0,2,2)]
    assert collision.is_collision((3, 3), obs) is False

def test_seg_collision():
    obs = [box(0,0,60,60)]
    