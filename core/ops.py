"""
This module defines common operations involving intervals
"""
from enum import Enum

from interval import interval


class PropertySatisfied(Enum):
    No = 0
    Yes = 1
    Maybe = 2


def interval_from_value(value: float, epsilon: float):
    """Procedure to create an interval from a single value"""
    return interval[value - epsilon, value + epsilon]


def max_upper(l: list):
    m = -10000
    for x in l:
        if x[0].sup > m:
            m = x[0].sup
    return m


def get_positive(a: list):
    """Procedure to extract the positive part of a matrix"""
    result = [[0 for _ in range(len(a[0]))] for _ in range(len(a))]

    for i in range(len(a)):
        for j in range(len(a[i])):
            if a[i][j][0].sup > 0 > a[i][j][0].inf:
                result[i][j] = interval[0, a[i][j][0].sup]
            else:
                result[i][j] = a[i][j] if a[i][j][0].inf > 0 else interval[0.0, 0.0]

    return result


def get_negative(a: list):
    """Procedure to extract the negative part of a matrix"""
    result = [[0 for _ in range(len(a[0]))] for _ in range(len(a))]

    for i in range(len(a)):
        for j in range(len(a[i])):
            if a[i][j][0].sup > 0 > a[i][j][0].inf:
                result[i][j] = interval[a[i][j][0].inf, 0]
            else:
                result[i][j] = a[i][j] if a[i][j][0].sup < 0 else interval[0.0, 0.0]

    return result


def add(a: list, b: list, *args):
    """Procedure to sum lists of intervals"""
    result = []

    for i in range(len(a)):
        result.append(a[i][0] + b[i][0])
        for x in args:
            result[i] += x[i][0]

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
                s += a[i][k] * b[k]
            result[i][j] = s

    return result


def create_disjunction_matrix(n_outs: int, label: int):
    """Procedure to create the matrix of the output property"""
    matrix = []
    c = 0
    for i in range(n_outs):
        if i != label:
            matrix.append([interval[0.0] for _ in range(n_outs)])
            matrix[c][label] = interval[1.0]
            matrix[c][i] = interval[-1.0]
            c += 1

    return matrix


def compute_max(weights: list, bounds: dict):
    """Procedure to compute the max value of the bounds through a linear transformation"""
    # The result is a list with one element and I return the element directly
    return add(matmul(get_positive(weights), bounds['lower']), matmul(get_negative(weights), bounds['upper']))[0]


def compute_min(weights: list, bounds: dict):
    """Procedure to compute the min value of the bounds through a linear transformation"""
    # The result is a list with one element and I return the element directly
    return add(matmul(get_positive(weights), bounds['upper']), matmul(get_negative(weights), bounds['lower']))[0]


def check_unsafe(bounds: dict, matrix: list, epsilon: float):
    """Procedure to check whether the output bounds are unsafe"""

    possible_counter_example = False

    disj_res = check_satisfied(bounds, matrix, epsilon)

    if disj_res == PropertySatisfied.Yes:
        # We are 100% sure there is a counter-example.
        # It can be any point from the input space.
        # Return anything from the input bounds
        # input_bounds = nn_bounds.numeric_pre_bounds[nn.get_first_node().identifier]
        return True  # , list(input_bounds.get_lower())
    elif disj_res == PropertySatisfied.Maybe:
        # We are not 100% sure there is a counter-example.
        # Call an LP solver when we need a counter-example
        possible_counter_example = True
        print('Maybe unsafe')
    else:  # disj_res == PropertySatisfied.No
        # nothing to be done. Maybe other disjuncts will be satisfied
        pass

    # At least for one disjunct there is a possibility of a counter-example.
    # Do a more powerful check with an LP solver
    if possible_counter_example:
        return False  # intersect_abstract_milp(star, nn, nn_bounds, prop)

        # Every disjunction is definitely not satisfied.
        # So we return False.
    return False  # , []


def check_satisfied(bounds: dict, matrix: list, epsilon: float):
    """
    Checks if the bounds satisfy the conjunction of constraints given by

        matrix * x <= 0

    Returns
    -------
    Yes if definitely satisfied
    No if definitely not satisfied
    Maybe when unsure
    """

    max_value = compute_max(matrix, bounds)
    min_value = compute_min(matrix, bounds)

    if min_value[0].inf > epsilon:
        # the constraint j is definitely not satisfied, as it should be <= 0
        return PropertySatisfied.No
    elif max_value[0].sup > epsilon:
        # the constraint j might not be satisfied, but we are not sure
        return PropertySatisfied.Maybe
    else:
        # if we reached here, means that all max values were below 0
        # so we now for sure that the property was satisfied
        # and there is a counter-example (any point from the input bounds)
        return PropertySatisfied.Yes
