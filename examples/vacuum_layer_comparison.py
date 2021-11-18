"""
An example showing the accuracy improvements of immersed boundary methods
vs a conventional staircased vacuum formulation.

Takes keyword arguments at the command line to configure.

Parameters
----------
mode : str
    If 'compare' then a plot comparing the wavefield and gathers for vacuum
    layer and immersed boundary topography is produced. The grids of each can
    be refined by passing parameters 'v_ref' and 'i_ref' as kwargs, where a
    value of 1 corresponds to the default resolution. 
"""

import sys
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from devito import TimeFunction, Eq, Operator, Function
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
    interior = get_interior(sdf.sdf, tuple(src.coordinates.data[0]), qc=True)

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

    midpoint = model.grid.shape[1]//2

    return rec.data.copy(), u.data[-1, 10:-10, midpoint, 10:-10].T.copy()


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
                               interior_point=tuple(src.coordinates.data[0]),
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

    midpoint = model.grid.shape[1]//2

    return rec.data.copy(), u.data[-1, 10:-10, midpoint, 10:-10].T.copy()


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
    rec.coordinates.data[:, 2] = 480 + 480*(np.sin(np.pi*np.linspace(100, 1100, num=101)/1200)) - 20

    return src, rec


def compare_shots(s_o, v_refinement=1., i_refinement=1.):
    """
    Compare the immersed boundary and vacuum layer for a given formulation

    Parameters
    ----------
    s_o : int
        The space order of the functions.
    v_refinement : float
        Refinement of the vacuum implementation relative to the baseline.
    i_refinement : float
        Refinement of the immersed boundary implementation relative to the
        baseline.
    """
    # Number of grid point (nx, ny, nz)
    vshape = tuple([int(1+100*v_refinement) for dim in range(3)])
    # Grid spacing in m. The domain size is 1x1x1km
    vspacing = tuple([1000/(vshape[0]-1) for dim in range(3)])
    # Needs to account for damping layers
    # FIXME: There must be a cleaner way to do this
    vorigin = tuple([v_refinement*10*vspacing[0] for dim in range(2)] + [0.])

    ishape = tuple([int(1+100*i_refinement) for dim in range(3)])
    ispacing = tuple([1000/(ishape[0]-1) for dim in range(3)])
    iorigin = tuple([i_refinement*10*ispacing[0] for dim in range(2)] + [0.])

    v = 1.5

    imodel = Model(vp=v, origin=iorigin, shape=ishape, spacing=ispacing,
                   space_order=s_o, nbl=10, bcs="damp")

    vmodel = Model(vp=v, origin=vorigin, shape=vshape, spacing=vspacing,
                   space_order=s_o, nbl=10, bcs="damp")

    t0 = 0.  # Simulation starts at t=0
    tn = 450.  # Simulation last 0.45 seconds (450 ms)
    dt = 0.6038  # Hardcoded timestep to keep stable

    time_range = TimeAxis(start=t0, stop=tn, step=dt)

    f0 = 0.015  # Source peak frequency is 15Hz (0.015 kHz)

    src, rec = setup_srcrec(imodel, time_range, f0)
    vsrc, vrec = setup_srcrec(vmodel, time_range, f0)

    # Define the wavefield with the size of the model and the time dimension
    iu = TimeFunction(name="u", grid=imodel.grid, time_order=2, space_order=s_o,
                      coefficients='symbolic')
    vu = TimeFunction(name="u", grid=vmodel.grid, time_order=2, space_order=s_o,
                      coefficients='symbolic')

    v_shot, v_wavefield = vacuum_shot(vmodel, vu, time_range, dt, vsrc, vrec)

    i_shot, i_wavefield = ib_shot(imodel, iu, time_range, dt, src, rec)

    return v_shot, v_wavefield, i_shot, i_wavefield


def plot_comparison(s_o, v_refinement=1., i_refinement=1.):
    """
    Plot a comparison of immersed boundary and vacuum shots.
    """

    v_shot, v_wavefield, i_shot, i_wavefield = compare_shots(s_o,
                                                             v_refinement=v_refinement,
                                                             i_refinement=i_refinement)

    # Normalization factors
    norm_v_shot = np.amax(np.abs(v_shot))
    norm_i_shot = np.amax(np.abs(i_shot))
    norm_v_wavefield = np.amax(np.abs(v_wavefield))
    norm_i_wavefield = np.amax(np.abs(i_wavefield))

    gather_extent = (0, 100, 450, 0)
    wavefield_extent = (0, 1000, 0, 1000)

    fig = plt.figure(constrained_layout=True, figsize=(9, 10))
    subfigs = fig.subfigures(2, 1, wspace=0.07)
    axsUp = subfigs[0].subplots(1, 2, sharey=True)
    subfigs[0].set_facecolor('0.75')
    axsUp[0].imshow(v_shot/norm_v_shot, extent=gather_extent, aspect='auto', cmap='seismic', vmin=-0.01, vmax=0.01)
    axsUp[0].set_title('Vacuum layer')
    axsUp[0].set_xlabel('Receiver number')
    axsUp[0].set_ylabel('Time (ms)')
    axsUp[1].imshow(i_shot/norm_i_shot, extent=gather_extent, aspect='auto', cmap='seismic', vmin=-0.01, vmax=0.01)
    axsUp[1].set_title('Immersed boundary')
    axsUp[1].set_xlabel('Receiver number')
    subfigs[0].suptitle('Shot gathers', fontsize='x-large')

    axsDwn = subfigs[1].subplots(1, 2, sharey=True)
    subfigs[1].set_facecolor('0.75')
    axsDwn[0].imshow(v_wavefield/norm_v_wavefield, origin='lower', extent=wavefield_extent, cmap='seismic', vmin=-0.15, vmax=0.15)
    axsDwn[0].plot(np.linspace(0, 1000, num=101), 480 + 480*(np.sin(np.pi*np.linspace(100, 1100, num=101)/1200)), color='k')
    axsDwn[0].scatter([500], [900], color='r')
    axsDwn[0].scatter(np.linspace(0, 1000, num=11), 480 + 480*(np.sin(np.pi*np.linspace(100, 1100, num=11)/1200)) - 20, color='b')
    axsDwn[0].set_title('Vacuum layer')
    axsDwn[0].set_xlabel('x (m)')
    axsDwn[0].set_ylabel('z (m)')
    axsDwn[1].imshow(i_wavefield/norm_i_wavefield, origin='lower', extent=wavefield_extent, cmap='seismic', vmin=-0.15, vmax=0.15)
    axsDwn[1].plot(np.linspace(0, 1000, num=101), 480 + 480*(np.sin(np.pi*np.linspace(100, 1100, num=101)/1200)), color='k')
    axsDwn[1].scatter([500], [900], color='r')
    axsDwn[1].scatter(np.linspace(0, 1000, num=11), 480 + 480*(np.sin(np.pi*np.linspace(100, 1100, num=11)/1200)) - 20, color='b')
    axsDwn[1].set_title('Immersed boundary')
    axsDwn[1].set_xlabel('x (m)')
    subfigs[1].suptitle('Wavefields', fontsize='x-large')

    fig.suptitle('Comparing topography implementations', fontsize='xx-large')

    plt.show()


def plot_convergence(s_o):
    """
    Compare convergence tests for vacuum layer and immersed boundary
    implementations.
    """
    raise NotImplementedError


def main(kwargs):
    mode = kwargs.get('mode', 'compare')
    s_o = int(kwargs.get('s_o', 4))

    if mode == 'compare':
        v_refinement = float(kwargs.get('v_ref', 1))
        i_refinement = float(kwargs.get('i_ref', 1))
        plot_comparison(s_o, v_refinement=v_refinement, i_refinement=i_refinement)
    elif mode == 'converge':
        plot_convergence(s_o)
    else:
        raise ValueError("Invalid mode set: {}".format(mode))


if __name__ == '__main__':
    main(dict(arg.split('=') for arg in sys.argv[1:]))  # Get kwargs
