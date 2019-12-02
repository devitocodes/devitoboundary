# Testbed file for understanding functionality of topography.py
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

from opesciboundary import Boundary
from devito import Grid, TimeFunction, Eq, solve, Operator, Substitutions, Coefficient
from examples.seismic import TimeAxis, RickerSource
from scipy.interpolate import interp2d
from sympy import finite_diff_weights

# Topography config
SUBSAMPLE = 5

grid = Grid(extent=(1000, 1000, 1000), shape=(101, 101, 101)) # Probably way too big

VP = 1.2

t0 = 0.  # Simulation starts a t=0
tn = 200.  # Simulation last 1 second (1000 ms)
dt = 0.2*grid.spacing[0]/(100*VP)  # Time step from model grid spacing

steps = int((t0+tn)/dt)+2

u = TimeFunction(name='u', grid=grid, time_order=2,
                space_order=4, coefficients='symbolic')

boundary_x = np.linspace(0, grid.extent[0], int(grid.shape[0]/SUBSAMPLE))

boundary_y = np.linspace(0, grid.extent[1], int(grid.shape[1]/SUBSAMPLE))

def dome_func(x, y):
    dome_z = 1.0*grid.extent[2] - np.sqrt(np.power(x-(grid.extent[0]/2), 2)
                                          + np.power(y-(grid.extent[1]/2), 2))
    dome_z[dome_z > 0.8*grid.extent[2]] = 0.8*grid.extent[2]
    return dome_z

mesh_x, mesh_y = np.meshgrid(boundary_x, boundary_y)

boundary_z = dome_func(mesh_x, mesh_y)

boundary_data = pd.DataFrame({'x':mesh_x.flatten(), 'y':mesh_y.flatten(), 'z':boundary_z.flatten()})

boundary_obj = Boundary(u, boundary_data, 2)

boundary_obj.plot_nodes()

time_range = TimeAxis(start=t0, stop=tn, step=dt)

f0 = 0.100
src = RickerSource(name='src', grid=grid, f0=f0,
                   npoint=1, time_range=time_range)

# First, position source centrally in all dimensions, then set depth
src.coordinates.data[0, :] = 500
src.coordinates.data[0, -1] = 250

std_coeffs = finite_diff_weights(2, range(-2, 3), 0)[-1][-1]
std_coeffs = np.array(std_coeffs)

x, y, z = grid.dimensions
x_coeffs = Coefficient(2, u, x, std_coeffs)
y_coeffs = Coefficient(2, u, y, std_coeffs)
z_coeffs = Coefficient(2, u, z, std_coeffs)
subs = Substitutions(x_coeffs, y_coeffs, z_coeffs)

# We can now write the PDE
pde = (1/VP)*u.dt2-u.laplace
eq = Eq(pde, 0, coefficients=boundary_obj.subs)

stencil = solve(eq.evaluate, u.forward)
src_term = src.inject(field=u.forward, expr=src*VP*dt**2)
op = Operator([Eq(u.forward, stencil)] + src_term)
op.apply(dt=dt)

plt.imshow(np.swapaxes(u.data[-1,:,20,:], 0, 1), origin="upper")
plt.colorbar()
plt.show()
