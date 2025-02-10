"""
This module defines the behavior of an interval arithmetic-based representation
of a neural network linear layer

"""


class IntervalLayer:
    pass


class LinearIntervalLayer(IntervalLayer):
    def __init__(self, weight: list, bias: list):
        super().__init__()

        self.weight = weight
        self.bias = bias
