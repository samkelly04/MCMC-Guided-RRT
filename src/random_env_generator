from shapely import box
from shapely.geometry import Polygon
import random

c_space = box(0, 0, 100, 100)
c_free = c_space

def rand_env_generator(c_space):
    obstacles = [ ] 
    N = random.randint(2, 5)
    obs_areas = [random.randint(4, 10) for _ in range(N)]

    while len(obstacles) < N:
        area = random.rantint(4, 10)
        width = random.randint(1, area)
        height = area / width

        for i in range(1, 500):
            x_min = random.uniform(0, 100 - width)
            y_min = random.uniform(0, 100 - height)
            candidate = box(x_min, y_min, x_min + width, y_min - height)

            if all(candidate.disjoint(obs["box"]) for obs in obstacles):
                obstacles.append({
                     "area": area,
                     "width": width,
                       "height": height,
                       "x_min": x_min,
                       "y_min": y_min,
                       "box": candidate
                       }) 
                break   





