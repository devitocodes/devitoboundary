from devito import Grid
from devitoboundary import PolyMesh
import numpy as np
# from scipy.spatial import KDTree
# from sys import setrecursionlimit

np.seterr(all='raise')
# setrecursionlimit(10000)

SUBSAMPLE = 5

grid = Grid(extent=(1000, 1000, 1000), shape=(101, 101, 101))

boundary_x = np.linspace(0,
                         grid.extent[0],
                         int(grid.shape[0]/SUBSAMPLE))

boundary_y = np.linspace(0,
                         grid.extent[1],
                         int(grid.shape[1]/SUBSAMPLE))


def inv_dome_func(x, y):
    dome_z = np.sqrt(np.power(x - (grid.extent[0]/2), 2)
                     + np.power(y - (grid.extent[1]/2), 2))
    dome_z[dome_z < 0.146*grid.extent[2]] = 0.146*grid.extent[2]
    return dome_z


mesh_x, mesh_y = np.meshgrid(boundary_x, boundary_y)

boundary_z = inv_dome_func(mesh_x, mesh_y)

# mesh = PolyMesh(data, grid)
data = np.vstack((mesh_x.flatten(), mesh_y.flatten(), boundary_z.flatten())).T
# print(len(data))
mesh = PolyMesh(data, grid)
