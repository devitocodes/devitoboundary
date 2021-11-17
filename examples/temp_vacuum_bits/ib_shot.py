import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import sympy as sp
from devito import Grid, TimeFunction, NODE, Eq, Operator, \
    Function, Coefficient, Substitutions, ConditionalDimension, Le
from devitoboundary import ImmersedBoundary, BoundaryConditions, \
    AxialDistanceFunction
from examples.seismic import TimeAxis, RickerSource, Model, Receiver


def ib_shot(model, time_range, f0):
    """
    Produce a shot with an immersed boundary
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

    infile = 'test_sinusoid_lo_res.ply'

    # Zero even derivatives on the boundary
    spec = {2*i: 0 for i in range(u.space_order)}
    bcs_u = BoundaryConditions(spec, u.space_order)

    functions = pd.DataFrame({'function': [u],
                              'bcs': [bcs_u],
                              'subs_function': [None],
                              'cautious': [False]},
                             columns=['function', 'bcs', 'subs_function', 'cautious'])

    # Create the immersed boundary surface
    surface = ImmersedBoundary('topography', infile, functions,
                               interior_point=(500, 500, 20),
                               qc=True, toggle_normals=True)
    # Configure derivative needed
    derivs = pd.DataFrame({'function': [u],
                           'derivative': [2],
                           'eval_offset': [(0., 0., 0.)]},
                          columns=['function', 'derivative', 'eval_offset'])
    coeffs = surface.subs(derivs)

    # We can now write the PDE
    stencil = Eq(u.forward,
                 2*u - u.backward + dt**2*v**2*u.laplace - model.damp * u.dt,
                 coefficients=coeffs)

    # Finally we define the source injection and receiver read function
    src_term = src.inject(field=u.forward,
                          expr=src)

    # Create interpolation expression for receivers
    rec_term = rec.interpolate(expr=u.forward)

    op = Operator([stencil] + src_term + rec_term)
    op(time=time_range.num-1, dt=0.6038)
    # op(time=time_range.num-1, dt=model.critical_dt)

    plt.imshow(u.data[-1, :, 50, :].T)
    plt.title("Pressure")
    plt.colorbar()
    plt.show()

    outfile = 'data/ib_wavefield.npy'
    np.save(outfile, u.data[-1, :, 50, :].T)

    return rec.data


# Define a physical size
shape = (101, 101, 101)  # Number of grid point (nx, ny, nz)
spacing = (10., 10., 10.)  # Grid spacing in m. The domain size is 1x1x1km
origin = (100., 100., 0.)

v = 1.5

model = Model(vp=v, origin=origin, shape=shape, spacing=spacing,
              space_order=4, nbl=10, bcs="damp")

t0 = 0.  # Simulation starts a t=0
tn = 450.  # Simulation last 0.5 seconds (500 ms)
# dt = model.critical_dt  # Time step from model grid spacing
dt = 0.6038

time_range = TimeAxis(start=t0, stop=tn, step=dt)

f0 = 0.015  # Source peak frequency is 10Hz (0.010 kHz)

shot_record = ib_shot(model, time_range, f0)

plt.imshow(shot_record, aspect='auto', vmin=-0.1, vmax=0.1, cmap='seismic')
plt.colorbar()
plt.show()

outfile = 'data/ib_gather.npy'
np.save(outfile, shot_record)
