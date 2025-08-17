from shapely.geometry import Point, Polygon, LineString
import numpy as np
#Shapely.geometry.point is shapely's geometry object for a single (x,y) location gives access to all shapely methods
#Obstacles is a list of rectangular obstacles generated randomly in the configuration space 
def is_collision(point, obstacles):
    x, y = float(point[0], point[1])
    p = Point(x, y)
    return any (obs.contains(p) for obs in obstacles)


def point_is_free(point, obstacles):
    '''' Returns TRUE if we have a valid point in the configuration space'''
    return not is_collision(point, obstacles)

def segment_is_collision(a, b, obstacles):
    ax, ay = float(a[0]), float(a[1])
    bx, by = float(b[0]), float(b[1])

    seg = LineString([(ax, ay), (bx, by)])

    return any(obs.intersects(seg) for obs in obstacles) # Use Intersects instead of Contains because Contains returns TRUE IFF the entire line segment is in the obstacle

def segment_is_free(a, b, obstacles):
    return not segment_is_collision(a, b, obstacles)