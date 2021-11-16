# An example showing the accuracy improvements of immersed boundary methods
# vs a conventional staircased vacuum formulation

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
    # FIXME: want to pull dt from time_range
    # FIXME: want to be able to provide an order

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

    # FIXME: could pull this outside the functions
    # Prescribe even spacing for receivers along the x-axis
    rec.coordinates.data[:, 0] = np.linspace(100, 1100, num=101)
    rec.coordinates.data[:, 1] = 600.  # Centered on y axis
    # Receivers track the surface at 20m depth
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
    op(time=time_range.num-1, dt=dt)

    return rec.data, u.data[-1, :, 50, :].T


def ib_shot(model, time_range, f0):
    """
    Produce a shot with an immersed boundary
    """
    # FIXME: want to pull dt from time_range
    # FIXME: want to be able to provide an order

    v = 1.5

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
    # Receivers track the surface at 20m depth
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
    op(time=time_range.num-1, dt=dt)

    return rec.data, u.data[-1, :, 50, :].T
