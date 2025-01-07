import numpy as np

# Laplacian Matrix
fd_coeffs = {1 : [[-3/2, 2, -1/2], [-1/2, 0, 1/2], [1/2, -2, 3/2]], 2 : [[2, -5, 4, -1], [1, -2, 1], [-1, 4, -5, 2]]}
delta_powers = {1 : (1, 1, 1), 2: (3, 2, 3)}

def derivative_1d(points : int, delta : float, order = int):
    forward, central, backward = fd_coeffs[order]
    delta_power = delta_powers[order]

    mat = np.zeros((points, points))
    for i in range(1, points - 1): mat[i, i-1:i+ len(central) - 1] = [x / delta ** delta_power[1] for x in central]
    
    mat[0][0:len(forward)] = [x / delta ** delta_power[0] for x in forward]
    mat[-1][-len(backward):] = [x / delta ** delta_power[2] for x in backward]

    return mat

def derivative_2d(x_points : int, x_delta : float, y_points : int, y_delta : int, order = int):
    return np.kron(derivative_1d(y_points, y_delta, order), np.eye(x_points)) + np.kron(np.eye(y_points), derivative_1d(x_points, x_delta, order))

mat = np.kron(np.eye(10), derivative_1d(10, 1/3, 2))

from matplotlib import pyplot as plt

plt.spy(mat)


