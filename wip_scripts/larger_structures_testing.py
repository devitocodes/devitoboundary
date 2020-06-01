from devito import Grid
from devitoboundary import PolySurface
import numpy as np
import matplotlib.pyplot as plt
# from scipy.spatial import KDTree
from sys import setrecursionlimit

np.seterr(all='raise')
setrecursionlimit(2000)

SUBSAMPLE = 5  # 5 is full size

grid = Grid(extent=(1000, 1000, 1000), shape=(176, 176, 176))

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


def realistic_func(x, y):
    surf_z = 0.5*grid.extent[2] + 0.05*grid.extent[2]*np.sin(2*np.pi*x/(1.3*grid.extent[0])) \
        + 0.05*grid.extent[2]*np.sin(2*np.pi*x/(0.23*grid.extent[0])) \
        + 0.025*grid.extent[2]*np.sin(2*np.pi*x/(0.56*grid.extent[0])) \
        + 0.025*grid.extent[2]*np.sin(2*np.pi*x/(0.09*grid.extent[0])) \
        + 0.05*grid.extent[2]*np.sin(2*np.pi*y/(0.23*grid.extent[1])) \
        + 0.05*grid.extent[2]*np.sin(2*np.pi*y/(0.84*grid.extent[1]))
    return surf_z


mesh_x, mesh_y = np.meshgrid(boundary_x, boundary_y)

# boundary_z = inv_dome_func(mesh_x, mesh_y)
boundary_z = realistic_func(mesh_x, mesh_y)

# mesh = PolySurface(data, grid)
data = np.vstack((mesh_x.flatten(), mesh_y.flatten(), boundary_z.flatten())).T
# print(len(data))
mesh = PolySurface(data, grid)
result = mesh.fd_node_sides()
print(result)
print(np.any(result))
print(np.count_nonzero(result))
plt.imshow(result[10].T)
plt.show()
