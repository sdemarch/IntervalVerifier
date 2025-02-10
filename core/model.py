"""
This module defines the behavior of an interval arithmetic-based representation
of a neural network model

"""

import pynever.strategies.conversion.representation as conv
from pynever import nodes
from pynever.strategies.conversion.converters.onnx import ONNXConverter

from core import ops
from core.linear import LinearIntervalLayer
from parser import vnnlib


class ModelOptions:
    pass


class IntervalModel:
    def __init__(self, onnx_path: str, work_precision: int = 64, options: ModelOptions = None):
        self.onnx_path = onnx_path
        self.work_precision = work_precision
        self.options = options

        self.layer = self.parse_layer()

    @staticmethod
    def check_robust(classifier_lbs: list[interval], classifier_ubs: list[interval], label: int) -> bool:
        """Procedure to check whether the robustness specification holds"""
        correct = classifier_lbs[label]

        return ops.max_upper(classifier_ubs) < correct.inf

    def parse_layer(self) -> LinearIntervalLayer:
        pynever_net = ONNXConverter().to_neural_network(conv.load_network_path(self.onnx_path))
        linear = pynever_net.get_roots()[0]

        assert len(pynever_net.nodes) == 1
        assert isinstance(linear, nodes.FullyConnectedNode)

        return LinearIntervalLayer(linear, self.work_precision)

    def propagate(self, lbs: list[interval], ubs: list[interval]) -> tuple[list[interval], list[interval]]:
        """Procedure to compute the numeric interval bounds of a linear layer"""
        weights_plus = ops.get_positive(self.layer.weight)
        weights_minus = ops.get_negative(self.layer.weight)

        low = ops.add(ops.matmul(weights_plus, lbs), ops.matmul(weights_minus, ubs), self.layer.bias)
        upp = ops.add(ops.matmul(weights_plus, ubs), ops.matmul(weights_minus, lbs), self.layer.bias)

        return low, upp

    def verify(self, vnnlib_path: str) -> bool:
        # 1: Read VNNLIB bounds
        in_lbs, in_ubs, label = vnnlib.read_vnnlib(vnnlib_path)

        # 2: Get interval input lbs and ubs
        in_lbs = [ops.interval_from_value(v, self.layer.epsilon) for v in in_lbs]
        in_ubs = [ops.interval_from_value(v, self.layer.epsilon) for v in in_ubs]

        # 3: Propagate input through linear layer
        out_lbs, out_ubs = self.propagate(in_lbs, in_ubs)

        # 4: Check output
        return IntervalModel.check_robust(out_lbs, out_ubs, label)
