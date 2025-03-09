"""
This module defines the behavior of an interval arithmetic-based representation
of a neural network linear layer

"""


class IntervalLayer:
    pass


class LinearIntervalLayer(IntervalLayer):
    """
    This class represents a Linear Layer where the weights and bias matrices
    contain interval values

    Attributes
    ----------
    weight : list[list[interval]] (n x m)
    bias : list[list[interval]] (n x 1)

    """

    def __init__(self, weight: list, bias: list):
        super().__init__()

        self.weight = weight
        self.bias = bias
