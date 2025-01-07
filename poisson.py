import numpy as np
from numpy import sin, cos
np.set_printoptions(linewidth=np.inf)

# L is x distance, H is y distance
L = 3
H = 3

delta = .1

x_intervals = int(L / delta)
y_intervals = int(H / delta)

key = 0

if key == 0:
    sinH = sin(H)
    sinL2 = sin(L ** 2) ** 2
    cos3L = cos(3 * L)

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
    if key == 0: return sin(x ** 2) ** 2 - cos(3 * x) * sinH
    return x * (1-x)

# lower x boundary (x = 0)
def g_1(y : float):
    if key == 0: return -1 * sin(y)
    return 0

# upper x boundary (x = L)
def g_2(y : float):
    if key == 0: return sinL2 - cos3L * sin(y)
    return 0

# creating the A' matrix
A_PRIME = np.identity(x_intervals - 1)

# setting the first and last rows manually
A_PRIME[0][0] = -4
A_PRIME[0][1] = 1
A_PRIME[-1][-1] = -4
A_PRIME[-1][-2] = 1

# setting all the intermediate row
for i in range(1, x_intervals - 2):
    curr = A_PRIME[i]
    curr[i-1] = 1
    curr[i] = -4
    curr[i+1] = 1

    A_PRIME[i] = curr

A_PRIME = np.array(A_PRIME)

# creating the identity and zero matricies
I = np.identity(x_intervals - 1)
Z = np.zeros((x_intervals - 1, x_intervals - 1))

# add the first row manually
first_row = [A_PRIME, I] + [Z] * (y_intervals -3)
A = [first_row]

# adding all intermediate rows
for i in range(1, y_intervals -2):
    curr_row = [Z for _ in range(y_intervals - 1)]
    curr_row[i -1] = I
    curr_row[i] = A_PRIME
    curr_row[i+1] = I
    A.append(curr_row)

# adding the last row of A manually
last_row = [Z] * (y_intervals -3) + [I, A_PRIME]
A.append(last_row)

# turning the list into a block matrix
A = np.block(A)

# range of x and y values, it starts and ends one delta away from 0 and L/H
x_range = np.linspace(delta, L - delta, num = x_intervals - 1)
y_range = np.linspace(delta, H - delta, num = y_intervals - 1)

from itertools import product

# creating a list of points by using all permutations of points
points = [(float(r[1]), float(r[0])) for r in product(x_range, y_range)]

# creating the Q vector
Q = np.array([Q_func(x,y) for x,y in points])

# creating the vector b
# B_PRIME is the blocks
B_PRIME = []
bptest = []
for y_i in y_range:
    B_PRIME.append([g_1(y_i)] + [0] * (x_intervals - 3) + [g_2(y_i)])
    bptest.append([f"g_1({y_i})"] + ["0"] * (x_intervals - 3) + [f"g_2({y_i})"])

# print(B_PRIME)

# adding f_1(x_i) to every element in the first vector, and f_2(x_i) to every element in the vector
for i, x_i in enumerate(x_range):
    B_PRIME[0][i] += f_1(x_i)
    bptest[0][i] += f" + f_1({x_i})"
    B_PRIME[-1][i] += f_2(x_i)
    bptest[-1][i] += f" + f_2({x_i})"

# np.block() does not work bc it needs to be only one row, not a matrix
B = []
for i in B_PRIME: B += i

# print(x_intervals)
# print(y_intervals)
btest = []
for i in bptest: btest += i
# print(np.array(btest))
# print(np.array(bptest))


# solving Av + b = (dx ** 2) * q for v
v = (delta ** 2 * Q)
v = np.subtract(v, B)
v = np.dot(np.linalg.inv(A), v)

v = list(v)

# defining all my x and y points, so I can create ordered triples with values from v
x_seq = [point[0] for point in points]
y_seq = [point[1] for point in points]

y_boundary_points = np.linspace(0, H, num = y_intervals + 1)
x_boundary_points = np.linspace(0, L, num = x_intervals + 1)

for y in y_boundary_points:
    x_seq.append(0)
    y_seq.append(y)
    v.append(g_1(y))

    x_seq.append(L)
    y_seq.append(y)
    v.append(g_2(y))

for x in x_boundary_points:
    x_seq.append(x)
    y_seq.append(0)
    v.append(f_1(x))

    x_seq.append(x)
    y_seq.append(H)
    v.append(f_2(x))

import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator

# plotting the data (is there a better way to do this?)
fig, ax = plt.subplots(subplot_kw={"projection": "3d"})

trisurf = ax.plot_trisurf(x_seq, y_seq, v,
                         cmap = plt.get_cmap('gist_gray'))
fig.colorbar(trisurf, ax = ax, shrink = 0.5, aspect = 5)  

# formatting the graph
ax.view_init(25, 45, 0)
ax.set_xlabel("x")
ax.set_ylabel("y")
ax.set_zlabel("z")

ax.xaxis.set_major_locator(MaxNLocator(5))
ax.yaxis.set_major_locator(MaxNLocator(6))
ax.zaxis.set_major_locator(MaxNLocator(5))

fig.tight_layout()

plt.show()