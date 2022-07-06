"""
AIF - Artificial Intelligence for Finance
Copyright (C) 2022 Christian Liguda

This program is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation, either version 3 of the License, or
any later version.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU General Public License for more details.

You should have received a copy of the GNU General Public License
along with this program.  If not, see <https://www.gnu.org/licenses/>.
"""

import math

import numba as nb
import numpy as np


class WeightedDistance:
    """The class is initialized with a weight vector that is used to calculate the weighted distance of two vectors."""

    def __init__(self, weights: np.ndarray):
        self.weights = weights

    def __call__(self, *args, **kwargs) -> float:
        x_1 = args[0]
        x_2 = args[1]
        assert len(self.weights) == len(x_1) == len(x_2), 'Len of vectors and weights does not match.'
        return weighted_distance(x_1, x_2, self.weights)


@nb.jit("f8(f8[:],f8[:],f8[:])", nopython=True)
def weighted_distance(x1, x2, weights):
    return math.sqrt(sum([weights[i] * (x1[i] - x2[i]) ** 2 for i in range(len(x1))]))
