"""
This module is to replicate some NumPy operations.
"""
import math


# Array operations
def linspace(start, stop, num=10):
    """Implementation of numpy.linspace."""
    if num < 2:
        return [start]
    step = (stop - start) / (num - 1)
    return [start + i*step for i in range(num)]


def multiply(vec, factor):
    return [factor * elem for elem in vec]


def add(*vecs):
    return [sum(elems) for elems in zip(*vecs)]


def subtract(vec1, vec2):
    return add(vec1, multiply(vec2, -1))


def norm(vec):
    return math.sqrt(sum([elem**2 for elem in vec]))


def dot(A, x):
    """Apply matrix `A` to vector `x`."""
    l, m, n = len(A), len(A[0]), len(x)
    if m != n:
        print('Shapes are incorrect.')
        return
    result = []
    for i in range(l):
        value = 0.
        for j in range(n):
            value += A[i][j] * x[j]
        result.append(value)
    return result


# Other functions
def step_func(x):
    "Heaviside step function."
    return x if x >= 0 else 0


def radians(angle_in_degrees):
    """Convert degrees to radians."""
    return angle_in_degrees * (math.pi / 180)


def degrees(angle_in_radians):
    """Convert radians to degrees."""
    return angle_in_radians * (180 / math.pi)
    
    
def put_angle_in_range(angle):
    """Put angle in range [0, 2pi]."""
    lim = 2 * math.pi
    if angle < 0:
        if abs(angle) > lim:
            angle = -(abs(angle) % lim)
        angle += lim
    if angle > lim:
        angle %= lim
    return angle