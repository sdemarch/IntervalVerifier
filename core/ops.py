"""
This module defines common operations involving intervals
"""

from interval import interval


def interval_from_value(value: float, epsilon: float):
    """Procedure to create an interval from a single value"""
    return interval[value - epsilon, value + epsilon]


def max_upper(l: list):
    m = -10000
    for x in l:
        if x.sup > m:
            m = x.sup
    return m


def get_positive(a: list):
    """Procedure to extract the positive part of a matrix"""
    result = []

    for i in range(len(a)):
        result.append([])
        for j in range(len(a[i])):
            if a[i][j].sup > 0 > a[i][j].inf:
                result[i][j] = interval[0, a[i][j].sup]
            else:
                result[i][j] = a[i][j] if a[i][j].inf > 0 else interval[0.0, 0.0]

    return result


def get_negative(a: list):
    """Procedure to extract the negative part of a matrix"""
    result = []

    for i in range(len(a)):
        result.append([])
        for j in range(len(a[i])):
            if a[i][j].sup > 0 > a[i][j].inf:
                result[i][j] = interval[a[i][j].inf, 0]
            else:
                result[i][j] = a[i][j] if a[i][j].sup < 0 else interval[0.0, 0.0]

    return result


def add(a: list, b: list, *args):
    """Procedure to sum lists of intervals"""
    result = []

    for i in range(len(a)):
        result.append(a[i] + b[i])
        for x in args:
            result[i] += x[i]

    return result


def matmul(a: list, b: list):
    """Procedure to multiply two 2-dimensional matrices"""

    assert len(a[0]) == len(b)

    n = len(a)
    m = len(b)
    q = len(b[0])

    result = [[0 for _ in range(q)] for _ in range(n)]

    for i in range(n):
        for j in range(q):
            s = 0
            for k in range(m):
                s += a[i][k] * b[k][j]
            result[i][j] = s

    return result
