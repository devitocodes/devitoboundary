# Testbed file for understanding functionality of topography.py
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

from opesciboundary import Boundary
from devito import Grid
from scipy.interpolate import interp2d

# Topography config
SUBSAMPLE = 0.2
VARIANCE = 0.3

grid = Grid(extent=(1, 1, 1), shape=(11, 11, 11))

# Randomly generated surface
seed_z = np.random.rand(grid.shape[0], grid.shape[1])

boundary_x = np.linspace(0, grid.extent[0], int(grid.shape[0]/SUBSAMPLE))
boundary_x += np.random.normal(scale=VARIANCE,
                               size=int(grid.shape[0]/SUBSAMPLE))
boundary_x[0] = 0
boundary_x[-1] = grid.extent[0]
boundary_x = boundary_x[boundary_x <= grid.extent[0]]
boundary_x = boundary_x[boundary_x >= 0]
boundary_x = np.sort(boundary_x)

boundary_y = np.linspace(0, grid.extent[1], int(grid.shape[1]/SUBSAMPLE))
boundary_y += np.random.normal(scale=VARIANCE,
                               size=int(grid.shape[1]/SUBSAMPLE))
boundary_y[0] = 0
boundary_y[-1] = grid.extent[1]
boundary_y = boundary_y[boundary_y <= grid.extent[1]]
boundary_y = boundary_y[boundary_y >= 0]
boundary_y = np.sort(boundary_y)

boundary_func = interp2d(np.linspace(0, grid.extent[0], grid.shape[0]),
                         np.linspace(0, grid.extent[1], grid.shape[1]),
                         seed_z)

boundary_z = boundary_func(boundary_x, boundary_y)

x, y = np.meshgrid(boundary_x, boundary_y)

boundary_data = pd.DataFrame({'x':x.flatten(), 'y':y.flatten(), 'z':boundary_z.flatten()})

boundary_obj = Boundary(grid, boundary_data)

#fig = plt.figure()
#ax = fig.add_subplot(111, projection='3d')
#ax.plot_surface(x, y, boundary_z)
#plt.show()
