"""
This module defines the behavior of an interval arithmetic-based representation
of a neural network model

"""

from core import ops
from core.layer import LinearIntervalLayer
from parser import onnx
from parser import vnnlib


class ModelOptions:
    pass


class IntervalModel:
    def __init__(self, onnx_path: str, work_precision: int = 64, options: ModelOptions = None):
        self.onnx_path = onnx_path
        self.work_precision = work_precision
        self.epsilon = 1e-6 if self.work_precision == 32 else 1e-12
        self.options = options

        self.layer = self.parse_layer()

    def check_robust(self, lbs: list, ubs: list, label: int) -> bool:
        """Procedure to check whether the robustness specification holds"""
        correct = lbs[label]
        others = list(ubs)
        others.pop(label)

        # Naive check - very unlikely
        if ops.max_upper(others) < correct[0].inf:
            return True
        else:
            n_disjunctions = len(others)

            # Create the matrices of the disjunctions
            matrix = ops.create_disjunction_matrix(len(lbs), label)
            bounds = {
                'lower': lbs,
                'upper': ubs
            }

            # For each disjunction in the output property, check none is satisfied by output_bounds.
            # If one disjunction is satisfied, then it represents a potential counter-example.
            for i in range(n_disjunctions):
                result = ops.check_unsafe(bounds, [matrix[i]], self.epsilon)

                if result:
                    return False

            return True

    def parse_layer(self) -> LinearIntervalLayer:
        """Procedure to read the first layer of a ONNX network"""
        nn = onnx.to_nn(self.onnx_path, self.epsilon)
        return nn[0]

    def propagate(self, lbs: list, ubs: list) -> tuple:
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
        in_lbs = [ops.interval_from_value(v, self.epsilon) for v in in_lbs]
        in_ubs = [ops.interval_from_value(v, self.epsilon) for v in in_ubs]

        # 3: Propagate input through linear layer
        out_lbs, out_ubs = self.propagate(in_lbs, in_ubs)

        # 4: Check output
        return self.check_robust(out_lbs, out_ubs, label)
