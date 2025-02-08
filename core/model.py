"""
This module defines the behavior of an interval arithmetic-based representation
of a neural network model

"""

import pynever.strategies.conversion.representation as conv
from pynever import nodes
from pynever.strategies.conversion.converters.onnx import ONNXConverter

from core.linear import LinearIntervalLayer


class ModelOptions:
    pass


class IntervalModel:
    def __init__(self, onnx_path: str, work_precision: int = 64, options: ModelOptions = None):
        self.onnx_path = onnx_path
        self.work_precision = work_precision
        self.options = options

        self.layer = self.parse_layer()

    def parse_layer(self) -> LinearIntervalLayer:
        pynever_net = ONNXConverter().to_neural_network(conv.load_network_path(self.onnx_path))
        linear = pynever_net.get_roots()[0]

        assert len(pynever_net.nodes) == 1
        assert isinstance(linear, nodes.FullyConnectedNode)

        return LinearIntervalLayer(linear, self.work_precision)

    def verify(self, vnnlib_path: str) -> bool:
        return self.onnx_path == '' and vnnlib_path == ''
