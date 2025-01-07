"""
Generates an approximate solution to the heat equation with homogenous dirichlet boundary condtions using finite differences.

When running this program, you may have to create a folder named 'tmp' folder as this file to store the graphs. You can change
the location where the graphs are stored at every call to the function 'gen_anim'. 
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import animation

from sympy import exp, sin, symbols
from sympy.utilities.lambdify import lambdify

np.set_printoptions(linewidth=np.inf)

# accuracy and range of x, y, and t
x_i, x_f = (0,1)
y_i, y_f = (0,1)
t_i, t_f = (0,1)

x_points = 10
y_points = 10
t_points = 200

# variables that will be used later
x_delta = (x_f - x_i) / (x_points - 1)
y_delta = (y_f - y_i) / (y_points - 1)
t_delta = (t_f - t_i) / (t_points - 1)

x = np.linspace(x_i, x_f, x_points)
y = np.linspace(y_i, y_f, y_points)
t = np.linspace(t_i, t_f, t_points)

nodes = x_points * y_points
nonboundary_nodes = (x_points - 2) * (y_points - 2)

# animation settings
duration = 5
animation_interval = 1000 * duration / t_points # t_points = frames
repeat_delay = 100000
cmap = 'Wistia'

# since the code for generating graphs is slow, turn this off if the actual
# solution graph already exists for a given range of x, y and t
regen_actual_solution = True

# generating function
x_symbol, y_symbol, t_symbol = symbols('x y t')

# heat equation: a * (d2u/dx2 + d2u/dy2) = du/dt
a = .1
scale_factor = 10

# CFL condition (not discussed in the paper, but this must be below .5 to get a stable graph)
print(a * t_delta / (x_delta ** 2) + a * t_delta / (y_delta ** 2))

xv, yv = np.meshgrid(x,y)

phi = scale_factor * sin(np.pi * x_symbol / x_f) * sin(np.pi * y_symbol / y_f) * exp(-t_symbol * a * (np.pi) ** 2 * (x_f ** -2 + y_f ** -2))
phi = np.vectorize(lambdify([x_symbol, y_symbol, t_symbol], phi))

actual_solution = np.array([phi(xv,yv, curr_t)for curr_t in t])

# code for generating animations
from matplotlib.ticker import MaxNLocator

def gen_anim(data, file_name):
    fig, ax = plt.subplots(1, 1, figsize = (6,6), subplot_kw={"projection": "3d"})

    max_z = np.max(data)
    min_z = np.min(data)

    def animate(i):
        ax.cla()
        trisurf = ax.plot_trisurf(xv.flatten(), yv.flatten(), data[i-1].flatten(), cmap = plt.get_cmap(cmap))
        ax.set_xlim((x_i, x_f))
        ax.set_ylim((y_i, y_f))
        ax.set_zlim((min_z, max_z))

    # formatting the graph
    ax.view_init(25, 45, 0)
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_zlabel("z")

    ax.xaxis.set_major_locator(MaxNLocator(5))
    ax.yaxis.set_major_locator(MaxNLocator(6))
    ax.zaxis.set_major_locator(MaxNLocator(5))

    fig.tight_layout()

    anim = animation.FuncAnimation(fig, animate, frames = t_points, interval = animation_interval, blit = False, repeat_delay = repeat_delay)
    anim.save(file_name)

if regen_actual_solution: gen_anim(actual_solution, 'tmp/actual_solution_2d.gif')

# conditions

# some parts may seem useless, but I'm planning on adding more customizability to the code
# and allowing for nonhomogenous and nonuniform boundary conditions.

# distribution at t = t_i
f_0 = scale_factor * sin(np.pi * x_symbol / x_f) * sin(np.pi * y_symbol / y_f)
f_0 = np.vectorize(lambdify([x_symbol, y_symbol], f_0))
init_distribution = f_0(xv, yv).flatten()

# lower boundary condition (x = x_i or y = y_i)
x_0 = 0
x_0 = np.vectorize(lambdify([t_symbol], x_0))

y_0 = 0
y_0 = np.vectorize(lambdify([t_symbol], y_0))

# upper boundary condtions (x = x_f or y = y_f)
x_L = 0
x_L = np.vectorize(lambdify([t_symbol], x_L))

y_L = 0
y_L = np.vectorize(lambdify([t_symbol], y_L))

# Laplacian Matrix
# inpsiration: https://www.petercheng.me/blog/discrete-laplacian-matrix
def laplacian_1d(points : int, delta : float):
    mat = np.zeros((points, points))
    for i in range(1, points - 1): mat[i, i-1:i+2] = (1,-2,1)
    mat *= delta
    
    mat[0][0:4] = (2,-5,4,-1)
    mat[-1][-4:] = (-1,4,-5,2)

    return mat / (delta ** 3)

A = np.kron(laplacian_1d(y_points, y_delta), np.eye(x_points)) + np.kron(np.eye(y_points), laplacian_1d(x_points, x_delta))

# function to set the boundary points
def bp(u):
    u[0:x_points] = y_0(x)
    u[-x_points:] = y_L(x)

    index = 0
    for x_val in x_0(y):
        u[index] = x_val
        index += x_points

    index = x_points - 1
    for x_val in x_L(y):
        u[index] = x_val
        index += x_points

    return u

# u_i,j_m+1 = u_i,j_m + a * dt * A @ u_m recursively
u = init_distribution
u_mat = np.zeros((t_points, x_points * y_points))
u_mat[0] = init_distribution

index = 0

for curr_index in t[:-1]:
    u += a * t_delta * A @ u
    u = bp(u)

    u_mat[index := index + 1] = u

gen_anim(u_mat, 'tmp/approximate_solution_2d.gif')

# graph of error term
actual_u = np.zeros((t_points, x_points * y_points))
for i in range(len(actual_solution)): actual_u[i] = actual_solution[i].flatten()

error_mat = actual_u - u_mat

gen_anim(error_mat, 'tmp/error_2d.gif')