"""
This module defines common operations involving intervals
"""

from interval import interval


def max_upper(l: list[interval]) -> float:
    m = -10000
    for x in l:
        if x.sup > m:
            m = x.sup
    return m


def interval_from_value(value: float, epsilon: float = 1e-6) -> interval:
    return interval[value - epsilon, value + epsilon]


def matmul(a: list[list[interval]], b: list[list[interval]]) -> list[list[interval]]:
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
