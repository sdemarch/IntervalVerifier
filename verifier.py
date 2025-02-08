"""
This is the entry point of the verifier
Usage: python verifier.py NETWORK.onnx PROPERTY.vnnlib [--precision BIT_PRECISION]

@author Stefano Demarchi

"""

from argparse import ArgumentParser

from core.model import IntervalModel

parser = ArgumentParser(description="Interval arithmetic-based neural networks verifier")
parser.add_argument('net', type=str, help='ONNX model file')
parser.add_argument('prop', type=str, help='VNNLIB property file')
parser.add_argument('--precision', type=int, default=32, help='Precision bits')

args = parser.parse_args()

model = IntervalModel(args.net, args.precision)
result = model.verify(args.prop)

print(result)
