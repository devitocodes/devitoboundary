# Testbed file for understanding functionality of topography.py
import numpy as np
import pandas as pd

from opesciboundary import Topography3D
from devito import Grid, TimeFunction
from scipy.interpolate import interp2d

# Topography config
SUBSAMPLE = 2
VARIANCE = 0.3

grid = Grid(extent=(1000, 1000, 1000), shape=(21, 21, 21))
function = TimeFunction(name='test_function', grid=grid, time_order=2,
                        space_order=4, coefficients='symbolic')

# Randomly generated surface
seed_z = np.random.rand(grid.shape[0], grid.shape[1])

boundary_x = np.linspace(0, grid.extent[0], int(grid.shape[0]/SUBSAMPLE))
boundary_x += np.random.normal(scale=VARIANCE*grid.spacing[0],
                               size=int(grid.shape[0]/SUBSAMPLE))
boundary_x[0] = 0
boundary_x[-1] = grid.extent[0]
boundary_x = boundary_x[boundary_x <= grid.extent[0]]
boundary_x = boundary_x[boundary_x >= 0]
boundary_x = np.sort(boundary_x)

boundary_y = np.linspace(0, grid.extent[1], int(grid.shape[1]/SUBSAMPLE))
boundary_y += np.random.normal(scale=VARIANCE*grid.spacing[1],
                               size=int(grid.shape[1]/SUBSAMPLE))
boundary_y[0] = 0
boundary_y[-1] = grid.extent[1]
boundary_y = boundary_y[boundary_y <= grid.extent[1]]
boundary_y = boundary_y[boundary_y >= 0]
boundary_y = np.sort(boundary_y)

boundary_func = interp2d(np.linspace(0, grid.extent[0], grid.shape[0]),
                         np.linspace(0, grid.extent[1], grid.shape[1]),
                         seed_z)


def dome_func(x, y):
    dome_z = 0.5*grid.extent[2] - np.sqrt(np.power(x-(grid.extent[0]/2), 2)
                                          + np.power(y-(grid.extent[1]/2), 2))
    dome_z[dome_z > 0.25*grid.extent[2]] = 0.25*grid.extent[2]
    return dome_z


def inv_dome_func(x, y):
    dome_z = np.sqrt(np.power(x-(grid.extent[0]/2), 2)
                     + np.power(y-(grid.extent[1]/2), 2))
    dome_z[dome_z < 0.296*grid.extent[2]] = 0.296*grid.extent[2]
    return dome_z


x, y = np.meshgrid(boundary_x, boundary_y)

boundary_z = inv_dome_func(x, y)

boundary_data = pd.DataFrame({'x': x.flatten(), 'y': y.flatten(), 'z': boundary_z.flatten()})

boundary_obj = Topography3D(function, boundary_data, 2, 0)

boundary_obj.plot_nodes(show_boundary=True)
