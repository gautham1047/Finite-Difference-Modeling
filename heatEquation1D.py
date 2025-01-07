import numpy as np
import numpy.linalg as npla
import matplotlib.pyplot as plt
from matplotlib import animation


from sympy import exp, sin, diff, symbols
from sympy.utilities.lambdify import lambdify

np.set_printoptions(linewidth=np.inf)

# accuracy and range of x and t
x_i, x_f = (-1,1)
t_i, t_f = (0,1)

x_points = 5
t_points = 5

x_delta = (x_f - x_i) / (x_points - 1)
t_delta = (t_f - t_i) / (t_points - 1)

x = np.linspace(x_i, x_f, x_points)
t = np.linspace(t_i, t_f, t_points)

# intial conditions
x_symbol, t_symbol = symbols('x t')

# heat distribution at t = t_i
f_0 = 10 / (1 + x_symbol ** 2)
f_0 = np.vectorize(lambdify([x_symbol], f_0))
initial_distribtion = f_0(x)

# x lower boundary condition
x_0 = 0
x_0 = np.vectorize(lambdify([t_symbol], x_0))

# x upper boundary condition
x_L = 0
x_L = np.vectorize(lambdify([t_symbol], x_L))

# heat equation: a * ∇^2 u = du/dt
a = .1

# creating the A matrix
r = a * t_delta / (x_delta ** 2)
rx = r / x_delta

A = np.zeros((x_points, x_points))

A[0, 0:4] = (1 + 2 * rx, -5 * rx, 4 * rx, -1 * rx)
for i in range(1, x_points - 1): A[i, i-1:i+2] = (r, 1 - 2 * r, r)
A[-1, -4:] = (-1 * rx, 4 * rx, -5 * rx, 1 + 2 * rx)

# prolongation and restriction matrices
P = np.zeros((x_points, x_points - 2))
P[1:-1,0:] = np.eye(x_points - 2)
R = P.transpose()

RAP_inv = npla.inv(R @ A @ P)

u = initial_distribtion
u_mat = np.zeros((t_points, x_points))
print(u_mat)
u_mat[0] = u

index = -1

for curr_time in t:
    curr_bc  = np.zeros(x_points)
    curr_bc[0] = x_0(curr_time)
    curr_bc[-1] = x_L(curr_time)

    u = RAP_inv @ (R @ (u - (A @ curr_bc)))
    u = P @ u + curr_bc

    u_mat[index := index + 1] = u

fig, ax = plt.subplots(1, 1, figsize = (6,6))

max_y = np.max(u_mat)
min_y = np.min(u_mat)

def animate(i):
    ax.cla()
    ax.plot(x, u_mat[i])
    ax.set_xlim((x_i, x_f))
    ax.set_ylim((min_y, max_y))

anim = animation.FuncAnimation(fig, animate, frames = t_points, interval = 1000, blit = False)
plt.show()