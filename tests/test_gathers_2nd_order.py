import pytest
import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from examples.seismic import Model, TimeAxis, RickerSource, Receiver
from devito import TimeFunction, Eq, solve, Operator
from devitoboundary import ImmersedBoundary, BoundaryConditions


def reference_shot(model, time_range, f0):
    """
    Produce a reference shot gather with a level, conventional free-surface
    implementation.
    """
    src = RickerSource(name='src', grid=model.grid, f0=f0,
                       npoint=1, time_range=time_range)

    # First, position source centrally in all dimensions, then set depth
    src.coordinates.data[0, :] = np.array(model.domain_size) * .5
    # Remember that 0, 0, 0 is top left corner
    # Depth is 100m from free-surface boundary
    src.coordinates.data[0, -1] = 600.

    # Create symbol for 101 receivers
    rec = Receiver(name='rec', grid=model.grid, npoint=101,
                   time_range=time_range)

    # Prescribe even spacing for receivers along the x-axis
    rec.coordinates.data[:, 0] = np.linspace(0, model.domain_size[0], num=101)
    rec.coordinates.data[:, 1] = 500.  # Centered on y axis
    rec.coordinates.data[:, 2] = 650.  # Depth is 150m from free surface

    # Define the wavefield with the size of the model and the time dimension
    u = TimeFunction(name="u", grid=model.grid, time_order=2, space_order=4)

    # We can now write the PDE
    pde = model.m * u.dt2 - u.laplace + model.damp * u.dt

    stencil = Eq(u.forward, solve(pde, u.forward))

    # Finally we define the source injection and receiver read function
    src_term = src.inject(field=u.forward,
                          expr=src * model.critical_dt**2 / model.m)

    # Create interpolation expression for receivers
    rec_term = rec.interpolate(expr=u.forward)

    x, y, z = model.grid.dimensions
    time = u.grid.stepping_dim
    # Throw a free surface in here
    free_surface_0 = Eq(u[time+1, x, y, 60], 0)
    free_surface_1 = Eq(u[time+1, x, y, 59], u[time+1, x, y, 61])
    free_surface_2 = Eq(u[time+1, x, y, 58], u[time+1, x, y, 62])
    free_surface = [free_surface_0, free_surface_1, free_surface_2]

    op = Operator([stencil] + src_term + rec_term + free_surface)

    op(time=time_range.num-1, dt=model.critical_dt)

    return rec.data


def tilted_shot(model, time_range, f0, tilt, qc=False, toggle_normals=False):
    """
    Produce a shot for the same setup, but tilted with immersed free surface
    """
    src = RickerSource(name='src', grid=model.grid, f0=f0,
                       npoint=1, time_range=time_range)

    # First, position source, then set depth
    src.coordinates.data[0, 0] = 500. - 100.*np.sin(np.radians(tilt))
    src.coordinates.data[0, 1] = 500.
    # Remember that 0, 0, 0 is top left corner
    # Depth is 100m from free-surface boundary
    src.coordinates.data[0, 2] = 500. + 100.*np.cos(np.radians(tilt))

    # Create symbol for 101 receivers
    rec = Receiver(name='rec', grid=model.grid, npoint=101,
                   time_range=time_range)

    # Prescribe even spacing for receivers along the x-axis
    rec_center_x = 500. - 150.*np.sin(np.radians(tilt))
    rec_center_z = 500. + 150.*np.cos(np.radians(tilt))

    rec_top_x = rec_center_x - 500.*np.cos(np.radians(tilt))
    rec_bottom_x = rec_center_x + 500.*np.cos(np.radians(tilt))

    rec_top_z = rec_center_z - 500.*np.sin(np.radians(tilt))
    rec_bottom_z = rec_center_z + 500.*np.sin(np.radians(tilt))

    rec.coordinates.data[:, 0] = np.linspace(rec_top_x, rec_bottom_x, num=101)
    rec.coordinates.data[:, 1] = 500.  # Centered on y axis
    rec.coordinates.data[:, 2] = np.linspace(rec_top_z, rec_bottom_z, num=101)

    # Define the wavefield with the size of the model and the time dimension
    u = TimeFunction(name="u", grid=model.grid, time_order=2, space_order=4,
                     coefficients='symbolic')

    infile = 'tests/trial_surfaces/angled_surface_'+str(tilt)+'.ply'

    # Zero even derivatives on the boundary
    spec = {2*i: 0 for i in range(u.space_order)}
    bcs_u = BoundaryConditions(spec, u.space_order)

    functions = pd.DataFrame({'function': [u],
                              'bcs': [bcs_u]},
                             columns=['function', 'bcs'])

    # Create the immersed boundary surface
    surface = ImmersedBoundary('topography', infile, functions,
                               interior_point=tuple(src.coordinates.data[0]),
                               qc=qc, toggle_normals=toggle_normals)
    # Configure derivative needed
    derivs = pd.DataFrame({'function': [u],
                           'derivative': [2],
                           'eval_offset': [(0., 0., 0.)]},
                          columns=['function', 'derivative', 'eval_offset'])
    coeffs = surface.subs(derivs)

    # We can now write the PDE
    pde = model.m * u.dt2 - u.laplace + model.damp * u.dt

    stencil = Eq(u.forward, solve(pde, u.forward),
                 coefficients=coeffs)

    # Finally we define the source injection and receiver read function
    src_term = src.inject(field=u.forward,
                          expr=src * model.critical_dt**2 / model.m)

    # Create interpolation expression for receivers
    rec_term = rec.interpolate(expr=u.forward)

    op = Operator([stencil] + src_term + rec_term)
    op(time=time_range.num-1, dt=model.critical_dt)

    return rec.data


class TestGathers:
    """
    A class for testing the accuracy of gathers resulting from a reflection off
    the immersed boundary.
    """

    @pytest.mark.parametrize('spec', [(5, False), (10, True), (15, True),
                                      (20, True), (25, True), (30, True),
                                      (35, True), (40, True), (45, False)])
    def test_tilted_boundary(self, spec):
        """
        Check that gathers for a tilted boundary match those generated with a
        conventional horizontal free surface and the same geometry.
        """
        tilt, toggle_normals = spec
        max_thres = 0.09
        avg_thres = 0.006
        # Define a physical size
        shape = (101, 101, 101)  # Number of grid point (nx, ny, nz)
        spacing = (10., 10., 10.)  # Grid spacing in m. The domain size is 1x1x1km
        origin = (0., 0., 0.)

        v = 1.5

        model = Model(vp=v, origin=origin, shape=shape, spacing=spacing,
                      space_order=4, nbl=10, bcs="damp")

        t0 = 0.  # Simulation starts a t=0
        tn = 500.  # Simulation last 0.5 seconds (500 ms)
        dt = model.critical_dt  # Time step from model grid spacing

        time_range = TimeAxis(start=t0, stop=tn, step=dt)

        f0 = 0.010  # Source peak frequency is 10Hz (0.010 kHz)

        ref = reference_shot(model, time_range, f0)
        tilted = tilted_shot(model, time_range, f0, tilt,
                             toggle_normals=toggle_normals)

        assert np.amax(np.absolute(ref - tilted)) < max_thres
        assert np.mean(np.absolute(ref-tilted)) < avg_thres


def main(tilt, toggle_normals):
    """For troubleshooting the sectioning"""

    # Define a physical size
    shape = (101, 101, 101)  # Number of grid point (nx, ny, nz)
    spacing = (10., 10., 10.)  # Grid spacing in m. The domain size is 1x1x1km
    origin = (0., 0., 0.)

    v = 1.5

    model = Model(vp=v, origin=origin, shape=shape, spacing=spacing,
                  space_order=4, nbl=10, bcs="damp")

    t0 = 0.  # Simulation starts a t=0
    tn = 500.  # Simulation last 0.5 seconds (500 ms)
    dt = model.critical_dt  # Time step from model grid spacing

    time_range = TimeAxis(start=t0, stop=tn, step=dt)

    f0 = 0.010  # Source peak frequency is 10Hz (0.010 kHz)

    ref = reference_shot(model, time_range, f0)
    tilted = tilted_shot(model, time_range, f0, tilt, qc=True,
                         toggle_normals=toggle_normals)
    """
    plt.imshow(ref)
    plt.show()
    plt.imshow(tilted)
    plt.show()
    plt.imshow(ref-tilted)
    """


if __name__ == "__main__":
    tilt = int(sys.argv[1])
    toggle_normals = bool(int(sys.argv[2]))
    print(tilt)
    print(toggle_normals)
    main(tilt, toggle_normals)
