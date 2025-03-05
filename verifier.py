"""
This is the entry point of the verifier
Usage: python verifier.py NETWORK.onnx PROPERTY.vnnlib [--precision DIGITS]

@author Stefano Demarchi

"""
import time
from argparse import ArgumentParser

from core.model import IntervalModel

parser = ArgumentParser(description="Interval arithmetic-based neural networks verifier")
parser.add_argument('net', type=str, help='ONNX model file')
parser.add_argument('prop', type=str, help='VNNLIB property file')
parser.add_argument('--precision', type=int, default=3, help='Precision digits')

args = parser.parse_args()

if __name__ == '__main__':
    model = IntervalModel(args.net, args.precision)

    start = time.perf_counter()
    print(f'Property verified: {model.verify(args.prop)}')
    print(f'Elapsed time     : {time.perf_counter() - start:.4f}s')
