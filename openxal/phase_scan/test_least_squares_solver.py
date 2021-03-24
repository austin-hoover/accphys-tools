"""Test least-squares solver."""

import random
from lib.utils import least_squares


# Solve A.x = b
b = [-1, 0.2, 0.9, 2.1]
A = [[ 0.,  1.], [ 1.,  1.], [ 2.,  1.], [ 3.,  1.]]

n = len(A[0])
x0 = [random.random() for _ in range(n)]
lb = n * [-1000]
ub = n * [+1000]

x = least_squares(A, b, x0, lb, ub, verbose=2)
print 'x =', x