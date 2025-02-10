"""
This module defines the behavior of an interval arithmetic-based representation
of a neural network linear layer

"""

from interval import interval
from pynever import nodes
from pynever.tensors import Tensor

from core import ops


class LinearIntervalLayer:
    def __init__(self, fc: nodes.FullyConnectedNode, precision: int):
        self.ref_layer = fc
        self.precision = precision
        self.epsilon = 1e-6 if self.precision == 32 else 1e-12  # TODO fix

        self.weight = self.interval_convert(fc.weight)
        self.bias = self.interval_convert(fc.bias)

    def interval_convert(self, matrix: Tensor) -> list[list[interval]]:
        """Procedure to convert a Tensor to an interval matrix"""
        result = []

        for i in range(matrix.shape[0]):
            result.append([])
            for j in range(matrix.shape[1]):
                result[i][j] = ops.interval_from_value(matrix[i, j], self.epsilon)

        return result
