"""
TCoordinates is a container for pixel coordinates (X,Y).

X -------->

Y
|
|
V

Can be added or subtracted to/from other objects of the same type.
Better coordinate holder than an ugly tuple
"""

import numpy as np


class TCoordinate:

    def __init__(self, x, y):
        self.x = np.intc(x)
        self.y = np.intc(y)

    def __add__(self, other):
        x = self.x + other.x
        y = self.y + other.y
        return TCoordinate(x, y)

    def __sub__(self, other):
        x = self.x - other.y
        y = self.x - other.y
        return TCoordinate(x, y)

    def values(self):
        return self.x, self.y
