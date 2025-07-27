from shapely.geometry import Point, Polygon
#Shapely.geometry.point is shapely's geometry object for a single (x,y) location gives access to all shapely methods
#Obstacles is a list of rectangular obstacles generated randomly in the configuration space 
def is_collision(point, obstalces):
    p = Point(point)
    return any(obs.intersects(p) for obs in obstalces)
