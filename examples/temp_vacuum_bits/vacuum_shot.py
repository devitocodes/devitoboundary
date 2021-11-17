import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import sympy as sp
from devito import Grid, TimeFunction, NODE, Eq, Operator, \
    Function, Coefficient, Substitutions, ConditionalDimension, Le
from devitoboundary import ImmersedBoundary, BoundaryConditions, \
    SignedDistanceFunction
from devitoboundary.segmentation import get_interior
from examples.seismic import TimeAxis, RickerSource, Model, Receiver


def vacuum_shot(model, time_range, f0):
    """
    Produce a shot with a vacuum layer
    """
    src = RickerSource(name='src', grid=model.grid, f0=f0,
                       npoint=1, time_range=time_range)

    # First, position source, then set depth
    src.coordinates.data[0, 0] = 600.
    src.coordinates.data[0, 1] = 600.
    # Remember that 0, 0, 0 is top left corner
    # Depth is 100m from free-surface boundary
    src.coordinates.data[0, 2] = 900.

    # Create symbol for 101 receivers
    rec = Receiver(name='rec', grid=model.grid, npoint=101,
                   time_range=time_range)

    # Prescribe even spacing for receivers along the x-axis

    rec.coordinates.data[:, 0] = np.linspace(100, 1100, num=101)
    rec.coordinates.data[:, 1] = 600.  # Centered on y axis
    rec.coordinates.data[:, 2] = 480 + 480*(np.sin(np.pi*np.linspace(0, 1000, num=101)/1000)) - 20

    # Define the wavefield with the size of the model and the time dimension
    u = TimeFunction(name="u", grid=model.grid, time_order=2, space_order=4,
                     coefficients='symbolic')

    infile = 'test_sinusoid_hi_res.ply'

    sdf = SignedDistanceFunction(u, infile,
                                 toggle_normals=True)
    interior = get_interior(sdf.sdf, (500, 500, 20), qc=True)

    v = Function(name='v', grid=model.grid)
    v.data[interior == 1] = 1.5

    # We can now write the PDE
    stencil = Eq(u.forward,
                 2*u - u.backward + dt**2*v**2*u.laplace - model.damp * u.dt)

    # Finally we define the source injection and receiver read function
    src_term = src.inject(field=u.forward,
                          expr=src)

    # Create interpolation expression for receivers
    rec_term = rec.interpolate(expr=u.forward)

    op = Operator([stencil] + src_term + rec_term)
    # op(time=time_range.num-1, dt=model.critical_dt)
    op(time=time_range.num-1, dt=0.6038)

    plt.imshow(u.data[-1, :, 50, :].T)
    plt.title("Pressure")
    plt.colorbar()
    plt.show()
    outfile = 'data/vacuum_wavefield_fourfold.npy'
    np.save(outfile, u.data[-1, :, 50, :].T)

    return rec.data


# Define a physical size
shape = (401, 401, 401)  # Number of grid point (nx, ny, nz)
spacing = (2.5, 2.5, 2.5)  # Grid spacing in m. The domain size is 1x1x1km
origin = (100., 100., 0.)

model = Model(vp=1.5, origin=origin, shape=shape, spacing=spacing,
              space_order=4, nbl=10, bcs="damp")

t0 = 0.  # Simulation starts a t=0
tn = 450.  # Simulation last 0.5 seconds (400 ms)
# dt = model.critical_dt  # Time step from model grid spacing
dt = 0.6038
print("dt = ", dt)

time_range = TimeAxis(start=t0, stop=tn, step=dt)

f0 = 0.015  # Source peak frequency is 10Hz (0.010 kHz)

shot_record = vacuum_shot(model, time_range, f0)

plt.imshow(shot_record, aspect='auto', vmin=-0.1, vmax=0.1, cmap='seismic')
plt.colorbar()
plt.show()

outfile = 'data/vacuum_gather_fourfold.npy'
np.save(outfile, shot_record)
