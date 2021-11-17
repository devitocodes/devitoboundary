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


def vacuum_shot(model, u, time_range, dt, src, rec):
    """
    Produce a shot with a vacuum layer
    """

    infile = 'topography/test_sinusoid_lo_res.ply'

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

    return rec.data.copy(), u.data[-1, :, 50, :].T.copy()


def ib_shot(model, u, time_range, dt, src, rec):
    """
    Produce a shot with an immersed boundary
    """

    infile = 'topography/test_sinusoid_lo_res.ply'

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

    v = 1.5

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

    return rec.data.copy(), u.data[-1, :, 50, :].T.copy()


def setup_srcrec(model, time_range, f0):
    """Return a ricker source and receivers"""
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

    return src, rec


def compare_shots(s_o, v_refinement=1.):
    """
    Compare the immersed boundary and vacuum layer for a given formulation

    Parameters
    ----------
    s_o : int
        The space order of the functions.
    v_refinement : float
        Refinement of the vacuum implementation relative to the immersed
        boundary implementation.
    """
    # FIXME: Add the option to refine in due course
    # FIXME: Add separate options to refine both immersed and vaccum formulations
    # Define a physical size
    shape = (101, 101, 101)  # Number of grid point (nx, ny, nz)
    spacing = (10., 10., 10.)  # Grid spacing in m. The domain size is 1x1x1km
    origin = (100., 100., 0.)  # Needs to account for damping layers

    vshape = tuple([int(1+100*v_refinement) for dim in range(3)])
    vspacing = tuple([1000/(vshape[0]-1) for dim in range(3)])
    vorigin = tuple([10*vspacing[0] for dim in range(3)])

    print("V shape", vshape)
    print("V spacing", vspacing)
    print("V origin", vorigin)

    v = 1.5

    model = Model(vp=v, origin=origin, shape=shape, spacing=spacing,
                  space_order=s_o, nbl=10, bcs="damp")

    vmodel = Model(vp=v, origin=vorigin, shape=vshape, spacing=vspacing,
                   space_order=s_o, nbl=10, bcs="damp")

    t0 = 0.  # Simulation starts a t=0
    tn = 400.  # Simulation last 0.4 seconds (400 ms)
    dt = 0.6038  # Hardcoded timestep to keep stable

    time_range = TimeAxis(start=t0, stop=tn, step=dt)

    f0 = 0.015  # Source peak frequency is 15Hz (0.015 kHz)

    src, rec = setup_srcrec(model, time_range, f0)
    vsrc, vrec = setup_srcrec(vmodel, time_range, f0)

    # Define the wavefield with the size of the model and the time dimension
    u = TimeFunction(name="u", grid=model.grid, time_order=2, space_order=s_o,
                     coefficients='symbolic')
    vu = TimeFunction(name="u", grid=vmodel.grid, time_order=2, space_order=s_o,
                      coefficients='symbolic')

    v_shot, v_wavefield = vacuum_shot(vmodel, vu, time_range, dt, vsrc, vrec)

    i_shot, i_wavefield = ib_shot(model, u, time_range, dt, src, rec)

    # FIXME: Need to tweak the min/max on these scales to highlight difference
    plt.imshow(v_shot/np.amax(np.abs(v_shot)), aspect='auto', cmap='seismic', vmin=-0.05, vmax=0.05)
    plt.show()

    plt.imshow(v_wavefield/np.amax(np.abs(v_wavefield)), cmap='seismic', vmin=-0.05, vmax=0.05, origin='lower')
    # FIXME: Add a line to show the boundary location
    plt.show()

    plt.imshow(i_shot/np.amax(np.abs(i_shot)), aspect='auto', cmap='seismic', vmin=-0.05, vmax=0.05)
    plt.show()

    plt.imshow(i_wavefield/np.amax(np.abs(i_wavefield)), cmap='seismic', vmin=-0.05, vmax=0.05, origin='lower')
    plt.show()


compare_shots(4, v_refinement=1.5)
