with open("tmp/delta_data.csv") as f:
    delta_data = []
    for line in f:
        delta_data.append(line)
    
import numpy as np
import matplotlib.pyplot as plt

delta_data = np.array(delta_data)

# accuracy and range of x, y, and t
x_i, x_f = (0,np.pi)
y_i, y_f = (0,np.pi)
t_i, t_f = (0,1)

L_x = x_f - x_i
L_y = y_f - y_i

x_points = 20
y_points = 20
t_points = 100

x = np.linspace(x_i, x_f, x_points)
y = np.linspace(y_i, y_f, y_points)
t = np.linspace(t_i, t_f, t_points)

xv, yv = np.meshgrid(x, y)

fig, ax = plt.subplots(1, 1, figsize = (6,6), subplot_kw={"projection": "3d"})
trisurf = ax.plot_trisurf(x, y, delta_data[-2])