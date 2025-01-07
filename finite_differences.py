# governing equation coeffs
# a_1 * y + a_2 * dy/dy + a_3 * d2y/dx2 = 0
# [y, dy/dy, d2y/dx2]
coeffs = [1,-4,5]

print(f"{coeffs[0]} * y + {coeffs[1]} * dy/dx + {coeffs[2]} * d2y/dx^2 = 0")

# graph is from 0 to 10
start = -11 
end = -1
dx = .019
num_points = int((end - start) / dx + 1)

# boundary points
y_i = -11
y_f = -1

# combining a_i with the delta_x term
# c_i = a_i / dx^i
coeffs = [coeff / (dx ** i) for i, coeff in enumerate(coeffs)]

# [y_(i-k)...y_i...y_(i+k)]
fd_eqs = [
    [0, 1, 0],
    [-1/2,0,1/2],
    [1,-2,1]
]

# e_0 * y_(i-k) + ... + e_k * y_i + ... + e_2k * y_(i+k)
eq_coeffs = [0 for _ in coeffs]

for coeff, fd_eq in zip(coeffs, fd_eqs):
    for i, fd_coeff in enumerate(fd_eq):
        eq_coeffs[i] += coeff * fd_coeff


# until here, this is made to be relatively easy to expand
# past this point, I had to assume that it was a second order ODE

import numpy as np

# formatting the arrays for testing
np.set_printoptions(linewidth=np.inf)

# creating the array using finite difference
# [  1,   0,...,      0,    0]
# [e_0, e_1,...,      0,    0]
# [         ...              ]
# [  0,   0,..., e_2k-1, e_2k]
# [  0,   0,...,      0,    1]

A = np.identity(num_points)

for i in range(1, num_points - 1):
    curr = A[i]
    curr[i-1] = eq_coeffs[0]
    curr[i] = eq_coeffs[1]
    curr[i+1] = eq_coeffs[2]

    A[i] = curr

A = np.array(A)

# [y_i, 0, 0, ... , y_f].T
B = np.array([0 for _ in range(num_points)]).T
B[0] = y_i
B[-1] = y_f

y = np.linalg.inv(A) * B.transpose()

# I'm not sure why I have to do this, but it works?
# For some reason y was a 21x21 matrix (which doesn't make sense since A.inv is 21x21 and B is 21x1)
y_list = [line[0] + line[-1] for line in y]

# generating a list of x values using the start and end values
x_list = []
i = start
while i <= end:
    x_list.append(i)
    i += dx

# graphing the data
import matplotlib.pyplot as plt
plt.plot(x_list, y_list)
plt.show()