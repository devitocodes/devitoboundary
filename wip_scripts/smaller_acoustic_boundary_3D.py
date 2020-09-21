import numpy as np
import matplotlib.pyplot as plt

from devitoboundary import ImmersedBoundarySurface
from devito import Grid, TimeFunction, Eq, ConditionalDimension, solve, Operator
from examples.seismic import TimeAxis, RickerSource
from sys import setrecursionlimit

# setrecursionlimit(3000)

# Topography config
SUBSAMPLE = 5  # 5
PMLS = 0

C = 0.01
VP = 1.2

# Goal is 126, 126, 126 (medium was 81, 81, 81)
grid = Grid(shape=(81, 81, 81), extent=(1000, 1000, 1000))

t0 = 0.  # Simulation starts at t=0
tn = 75.
# tn = 170.  # Simulation length in ms
dt = C*grid.spacing[0]/VP

steps = int((t0+tn)/dt)+2
nsnaps = 100  # 50
factor = round(steps / nsnaps)

time_subsampled = ConditionalDimension(
    't_sub', parent=grid.time_dim, factor=factor)
usave = TimeFunction(name='usave', grid=grid, time_order=2, space_order=2,
                     save=(steps + factor - 1) // factor, time_dim=time_subsampled)

u = TimeFunction(name='u', grid=grid, time_order=2,
                 space_order=4, coefficients='symbolic')

boundary_x = np.linspace(0, grid.extent[0], int(grid.shape[0]/SUBSAMPLE))

boundary_y = np.linspace(0, grid.extent[1], int(grid.shape[1]/SUBSAMPLE))


def inv_dome_func(x, y):
    dome_z = np.sqrt(np.power(x - (grid.extent[0]/2), 2)
                     + np.power(y - (grid.extent[1]/2), 2))
    dome_z[dome_z < 0.25*grid.extent[2]] = 0.25*grid.extent[2]
    return dome_z


def realistic_func(x, y):
    surf_z = 0.5*grid.extent[2] + 0.05*grid.extent[2]*np.sin(2*np.pi*x/(1.3*grid.extent[0])) \
        + 0.05*grid.extent[2]*np.sin(2*np.pi*x/(0.23*grid.extent[0])) \
        + 0.025*grid.extent[2]*np.sin(2*np.pi*x/(0.56*grid.extent[0])) \
        + 0.025*grid.extent[2]*np.sin(2*np.pi*x/(0.09*grid.extent[0])) \
        + 0.05*grid.extent[2]*np.sin(2*np.pi*y/(0.23*grid.extent[1])) \
        + 0.05*grid.extent[2]*np.sin(2*np.pi*y/(0.84*grid.extent[1]))
    return surf_z


surface_x, surface_y = np.meshgrid(boundary_x, boundary_y)

# surface_z = inv_dome_func(surface_x, surface_y)
surface_z = realistic_func(surface_x, surface_y)

surface_data = np.vstack((surface_x.flatten(), surface_y.flatten(), surface_z.flatten())).T

surface = ImmersedBoundarySurface(surface_data, (u,), stencil_file='test_cache.dat')

# Zero even derivatives on the boundary
bc_0 = Eq(surface.u(u, surface.x_b(u)), 0)
bc_2 = Eq(surface.u(u, surface.x_b(u), 2), 0)
bcs = [bc_0, bc_2]

surface.add_bcs(u, bcs)

surface.plot_boundary()

# boundary_obj.plot_nodes(save=True, save_path="images/boundary_plot")

time_range = TimeAxis(start=t0, stop=tn, step=dt)

f0 = 0.100  # 100Hz
src = RickerSource(name='src', grid=grid, f0=f0,
                   npoint=1, time_range=time_range)

# First, position source centrally in all dimensions, then set depth
src.coordinates.data[0, :] = 500
src.coordinates.data[0, -1] = 800

# Dictionary of derivatives needed
deriv = {u: 2}

# We can now write the PDE
pde = VP*u.dt2 - u.laplace
eq = Eq(pde, 0, coefficients=surface.subs(deriv))

stencil = solve(eq.evaluate, u.forward)
src_term = src.inject(field=u.forward, expr=src*dt**2/VP)
op = Operator([Eq(u.forward, stencil)] + [Eq(usave, u)] + src_term)
op.apply(dt=dt)

for i in range(nsnaps):
    fig = plt.figure()
    plt.imshow(np.swapaxes(usave.data[i, :, int(grid.shape[1]/2), :], 0, 1),
               origin="upper", extent=[0, grid.extent[0], grid.extent[2], 0],
               vmin=-0.005, vmax=0.005, cmap='seismic')
    plt.colorbar()
    plt.xlabel("x (m)")
    plt.ylabel("z (m)")
    plt.savefig("images/image-%s" % str(i))
    # plt.show()
    plt.close()
