import matplotlib.pyplot as plt
from matplotlib import animation
import numpy as np
from sympy import sin,cos,exp, symbols, lambdify

x_i, x_f = (0,20)
y_i, y_f = (0,20)
t_i, t_f = (0,2)

x_points = 21
y_points = 21
t_points = 201

x = np.linspace(x_i, x_f, x_points)
y = np.linspace(y_i, y_f, y_points)
t = np.linspace(t_i, t_f, t_points)

x_symbol, y_symbol, t_symbol = symbols('x y t')

u = np.vectorize(lambdify([x_symbol, y_symbol, t_symbol], -cos(x_symbol) * sin(y_symbol) * exp(-2 * t_symbol)))
v = np.vectorize(lambdify([x_symbol, y_symbol, t_symbol], sin(x_symbol) * cos(y_symbol) * exp(-2 * t_symbol)))

p = np.vectorize(lambdify([x_symbol, y_symbol, t_symbol], -(cos(2 * x_symbol) + cos(2 * y_symbol)) * exp(-4 * t_symbol) / 4))

xv, yv = np.meshgrid(x, y)

u_mat = np.array([u(xv, yv, curr_t) for curr_t in t])
v_mat = np.array([v(xv, yv, curr_t) for curr_t in t])
p_mat = np.array([p(xv, yv, curr_t) for curr_t in t])

magnitude = lambda x, y: np.sqrt(x ** 2 + y ** 2)
vfunc = np.vectorize(magnitude)

print(u_mat.shape)
print(v_mat.shape)
print(p_mat.shape)

mag = vfunc(u_mat, v_mat)

# animation settings
duration = 5
animation_interval = 1000 * duration / t_points # t_points = frames
repeat_delay = 100000
cmap = 'Wistia'

def gen_anim(data, file_name):
    fig, ax = plt.subplots(1, 1, figsize = (6,6))

    def animate(i):
        ax.cla()
        ax.pcolormesh(x, y, data[i][:-1,:-1], cmap = plt.get_cmap(cmap), shading='flat', vmin=data.min(), vmax=data.max())
        ax.set_xlim((x_i, x_f))
        ax.set_ylim((y_i, y_f))

    anim = animation.FuncAnimation(fig, animate, frames = t_points, interval = animation_interval, blit = False, repeat = False)
    anim.save(file_name)

gen_anim(mag, 'tmp/test_mag.gif')
gen_anim(p_mat, 'tmp/test_p.gif')
