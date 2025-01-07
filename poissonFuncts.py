# This file is for the purpoes of testing each aspect of the code individually


from poissonData import *

from itertools import product

def generate_A_prime(x_intervals : int):
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

    return np.array(A_PRIME)

def generate_A_matrix(x_intervals : int, y_intervals : int):
    A_PRIME = generate_A_prime(x_intervals)
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
    return np.block(A)

# range of x and y values, it starts and ends one delta away from 0 and L/H
x_range = np.linspace(delta, L - delta, num = x_intervals - 1)
y_range = np.linspace(delta, H - delta, num = y_intervals - 1)

# creating a list of points by using all permutations of points
points = [(float(r[1]), float(r[0])) for r in product(x_range, y_range)]

# creating the Q vector
def generate_Q():
    return np.array([Q_func(x,y) for x,y in points])

# creating the vector b
def generate_B():
    # B_PRIME is the blocks
    B_PRIME = [[g_1(y_i)] + [0] * (x_intervals - 3) + [g_2(y_i)] for y_i in y_range]
    # adding f_1(x_i) to every element in the first vector, and f_2(x_i) to every element in the vector
    for i, x_i in enumerate(x_range):
        B_PRIME[0][i] += f_1(x_i)
        B_PRIME[-1][i] += f_2(x_i)

    B = []
    for i in B_PRIME: B += i
    return B




