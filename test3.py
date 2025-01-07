import numpy as np
import numpy.linalg as npla
import matplotlib.pyplot as plt
from sys import maxsize

from sympy import exp, sin, diff, symbols
from sympy.utilities.lambdify import lambdify

np.set_printoptions(linewidth=np.inf)
np.set_printoptions(threshold=maxsize)

# defining range and precision
x_i, x_f = (0,2)
y_i, y_f = (0,2)

delta = .25

# number of points in x and y direction
x_points = int((x_f - x_i) // delta) + 1
y_points = int((y_f - y_i) // delta) + 1

x = np.linspace(x_i, x_f, x_points)
y = np.linspace(y_i, y_f, y_points)

nodes = x_points * y_points
nonboundary_nodes = (x_points - 2) * (y_points - 2)

A = np.zeros((nodes, nodes))

for i in range(nodes):
    if (i < x_points or i > nodes - x_points):
        A[i][i] = 1

print(A)