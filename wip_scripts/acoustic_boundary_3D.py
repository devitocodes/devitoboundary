# TODO: Test with smaller grid and thinner absorbing layers
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


from devitoboundary import Topography3D
from devito import TimeFunction, Eq, solve, Operator, ConditionalDimension
from examples.seismic import TimeAxis, RickerSource, Model

# Topography config
SUBSAMPLE = 5
PMLS = 50

VP = 1.2

model = Model(vp=VP, origin=(0, 0, 0), shape=(126, 126, 126),
              spacing=(8, 8, 8), space_order=4, nbl=PMLS)

t0 = 0.  # Simulation starts at t=0
tn = 170.  # Simulation length in ms
dt = model.critical_dt*0.01  # Time step from model grid spacing

steps = int((t0+tn)/dt)+2
nsnaps = 200
factor = round(steps / nsnaps)

time_subsampled = ConditionalDimension(
    't_sub', parent=model.grid.time_dim, factor=factor)
usave = TimeFunction(name='usave', grid=model.grid, time_order=2, space_order=2,
                     save=(steps + factor - 1) // factor, time_dim=time_subsampled)

u = TimeFunction(name='u', grid=model.grid, time_order=2,
                 space_order=4, coefficients='symbolic')

boundary_x = np.linspace(-PMLS*model.grid.spacing[0],
                         model.grid.extent[0] - PMLS*model.grid.spacing[0],
                         int(model.grid.shape[0]/SUBSAMPLE))

boundary_y = np.linspace(-PMLS*model.grid.spacing[1],
                         model.grid.extent[1] - PMLS*model.grid.spacing[1],
                         int(model.grid.shape[1]/SUBSAMPLE))


def dome_func(x, y):
    dome_z = 1.0*model.grid.extent[2] - np.sqrt(np.power(x-(model.grid.extent[0]/2), 2)
                                                + np.power(y-(model.grid.extent[1]/2), 2))
    dome_z[dome_z > 0.8*model.grid.extent[2]] = 0.8*model.grid.extent[2]
    return dome_z


def inv_dome_func(x, y):
    dome_z = np.sqrt(np.power(x + PMLS*model.grid.spacing[0] - (model.grid.extent[0]/2), 2)
                     + np.power(y + PMLS*model.grid.spacing[1] - (model.grid.extent[1]/2), 2))
    dome_z[dome_z < 0.146*model.grid.extent[2]] = 0.146*model.grid.extent[2]
    return dome_z


mesh_x, mesh_y = np.meshgrid(boundary_x, boundary_y)

boundary_z = inv_dome_func(mesh_x, mesh_y)

boundary_data = pd.DataFrame({'x': mesh_x.flatten(), 'y': mesh_y.flatten(), 'z': boundary_z.flatten()})

boundary_obj = Topography3D(u, boundary_data, 2, pmls=PMLS)

boundary_obj.plot_nodes(save=True, save_path="images/boundary_plot")

time_range = TimeAxis(start=t0, stop=tn, step=dt)

f0 = 0.100
src = RickerSource(name='src', grid=model.grid, f0=f0,
                   npoint=1, time_range=time_range)

# First, position source centrally in all dimensions, then set depth
src.coordinates.data[0, :] = 500
src.coordinates.data[0, -1] = 800


# We can now write the PDE
pde = model.m*u.dt2 - u.laplace + model.damp*u.dt
eq = Eq(pde, 0, coefficients=boundary_obj.subs)

stencil = solve(eq.evaluate, u.forward)
src_term = src.inject(field=u.forward, expr=src*dt**2/model.m)
op = Operator([Eq(u.forward, stencil)] + [Eq(usave, u)] + src_term)
op.apply(dt=dt)

for i in range(nsnaps):
    fig = plt.figure()
    plt.imshow(np.swapaxes(usave.data[i, PMLS:-PMLS, int(model.grid.shape[1]/2), PMLS:-PMLS], 0, 1),
               origin="upper", extent=[0, model.grid.extent[0] - 2*PMLS*model.grid.spacing[0], model.grid.extent[2] - 2*PMLS*model.grid.spacing[2], 0],
               vmin=-0.005, vmax=0.005)
    plt.colorbar()
    plt.xlabel("x (m)")
    plt.ylabel("z (m)")
    plt.savefig("images/image-%s" % str(i))
    plt.close()
