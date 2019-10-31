import numpy as np
from devito import Grid, Function, TimeFunction, Eq, Operator, solve
from devito import ConditionalDimension, Constant, first_derivative, second_derivative
from devito import left, right
from math import exp, floor

from devito.data import Decomposition

import matplotlib
import matplotlib.pyplot as plt
from matplotlib import cm
from examples.seismic.source import RickerSource, TimeAxis

from devito.boundary_object import Boundary


# Parameters
pi = np.pi
eps = np.finfo(float).eps


# Boundary function definitions
def boundary_func(*args):
    m = -1.1
    c = 1.05
    for arg in args:
        x = arg
    y = m*x+c
    return y

def iboundary_func(*args):
    m = -1.1
    c = 1.05
    for arg in args:
        y = arg
    x = (y-c)/m
    return x

# Set up the Devito grid
Lx = 1.
Ly = 1.

Nx = 101
Ny = Nx

dx = Lx/(Nx-1)
dy = dx

grid = Grid(shape=(Nx,Ny), extent=(Lx,Ly))
time = grid.time_dim
t = grid.stepping_dim
x,y = grid.dimensions

# Create our immersed boundar
boundary_obj = Boundary(grid, boundary_func, iboundary_func)

stencil_data = boundary_obj.stencil()

print(stencil_data)

# Plot of 'nodes' where the stencil is modified
# relative to the boundary
xg = np.linspace(0,Lx,Nx)
yg = xg

nodes = stencil_data["Node"].values
nnodes = len(stencil_data.index)

xn = np.zeros(nnodes)
yn = np.zeros(nnodes)

for j in range(0,nnodes):
    b = np.asarray(nodes[j])*[dx,dy]
    xn[j] = b[0]
    yn[j] = b[1]

yb = boundary_func(xg)

fig = plt.figure(figsize=(14, 7))
ax1 = fig.add_subplot(111)
ax1.plot(xg,yb,'-b')
ax1.plot(xn,yn,'.r')
ax1.axis([0, Lx, 0, Ly])
ax1.set_xlabel('$x$')
ax1.set_ylabel('$y$')
ax1.minorticks_on()
ax1.set_xticks(xg, minor=True)
ax1.grid(which='major', linestyle='-', linewidth='0.5', color='green')
ax1.grid(which='minor', linestyle=':', linewidth='0.5', color='black')
plt.show()
