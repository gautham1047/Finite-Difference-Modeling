import numpy as np
from numpy import sin, cos
np.set_printoptions(linewidth=np.inf)

# L is x distance, H is y distance
L = 1
H = 2

delta = .5

x_intervals = int(L / delta)
y_intervals = int(H / delta)

key = 0

# defining Q and boundary functions
def Q_func(x : float, y : float):
    if key == 0: return 10 * cos(3 * x) * sin(y) + 8 * x ** 2 * (cos(x ** 2) ** 2) - 8 * x ** 2 * (sin(x ** 2) ** 2) + 4 * cos(x ** 2) * sin(x ** 2)
    return -2 * y ** 2 + 2 * x * (1 - x)

# lower y boundary (y = 0)
def f_1(x : float):
    if key == 0: return sin(x ** 2) ** 2
    return 0

# upper y boundary (y = H)
def f_2(x : float):
    if key == 0: return sin(x ** 2) ** 2 - cos(3 ** x) * sin(H)
    return x * (1-x)

# lower x boundary (x = 0)
def g_1(y : float):
    if key == 0: return -1 * sin(y)
    return 0

# upper x boundary (x = L)
def g_2(y : float):
    if key == 0: return sin(L ** 2) ** 2 - cos(3 * L) * sin(y)
    return 0