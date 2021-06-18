from manimlib import *

class BraceBetweenPoints(Brace):
    def __init__(
        self,
        point_1,
        point_2,
        direction=ORIGIN,
        **kwargs
    ):
        if all(direction == ORIGIN):
            line_vector = np.array(point_2) - np.array(point_1)
            direction = np.array([line_vector[1], -line_vector[0], 0])
        Brace.__init__(self, Line(point_1, point_2), direction=direction, **kwargs)
