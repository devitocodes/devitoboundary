import numpy as np
import pandas as pd
import sympy as sp
import matplotlib.pyplot as plt
import pytest
import sys

from examples.seismic import Model, TimeAxis, RickerSource, Receiver
from devito import TimeFunction, Eq, Operator, NODE, ConditionalDimension, \
    Function, Le
from devitoboundary import BoundaryConditions, ImmersedBoundary, AxialDistanceFunction


def reference_shot(model, time_range, f0, s_o):
    """
    Produce a reference shot gather with a level, conventional free-surface
    implementation.
    """
    x, y, z = model.grid.dimensions
    t = model.grid.stepping_dim
    dt = model.critical_dt/2  # Time step from model grid spacing

    vel = 1.5
    rho = 1.5

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
    p = TimeFunction(name="p", grid=model.grid, time_order=1, space_order=s_o,
                     staggered=NODE)
    vx = TimeFunction(name="v_x", grid=model.grid, time_order=1, space_order=s_o,
                      staggered=x)
    vy = TimeFunction(name="v_y", grid=model.grid, time_order=1, space_order=s_o,
                      staggered=y)
    vz = TimeFunction(name="v_z", grid=model.grid, time_order=1, space_order=s_o,
                      staggered=z)

    # We can now write the system of PDEs
    eq_vx = Eq(vx.forward, vx - dt*p.dx/rho)
    eq_vy = Eq(vy.forward, vy - dt*p.dy/rho)
    eq_vz = Eq(vz.forward, vz - dt*p.dz/rho)
    eq_p = Eq(p.forward,
              p - dt*vel**2*rho*(vx.forward.dx + vy.forward.dy + vz.forward.dz)/(1+model.damp))

    # Finally we define the source injection and receiver read function
    src_term = src.inject(field=p.forward,
                          expr=src)

    # Create interpolation expression for receivers
    rec_term = rec.interpolate(expr=p.forward)

    # Pressure free surface conditions
    free_surface_p = [Eq(p[t+1, x, y, 60], 0)]
    for i in range(s_o//2):
        free_surface_p.append(Eq(p[t+1, x, y, 59-i], -p[t+1, x, y, 61+i]))

    free_surface_v = []
    for i in range(s_o//2):
        free_surface_v.append(Eq(vz[t+1, x, y, 59-i], vz[t+1, x, y, 60+i]))

    op = Operator([eq_vx, eq_vy, eq_vz] + free_surface_v
                  + [eq_p] + free_surface_p
                  + src_term + rec_term)

    op(time=time_range.num-1, dt=model.critical_dt/2)

    """
    plt.imshow(p.data[-1, :, 60].T, origin='upper')
    plt.colorbar()
    plt.show()
    """

    return rec.data


def tilted_shot(model, time_range, f0, s_o, tilt, qc=False, toggle_normals=False):
    """
    Produce a shot for the same setup, but tilted with immersed free surface
    """
    # FIXME: Something is wonky here -> am I using a consistent surface?
    x, y, z = model.grid.dimensions
    h_x, h_y, h_z = model.grid.spacing
    dt = model.critical_dt/2  # Time step from model grid spacing

    vel = 1.5
    rho = 1.

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
    p = TimeFunction(name="p", grid=model.grid, time_order=2, space_order=s_o,
                     staggered=NODE, coefficients='symbolic')
    vx = TimeFunction(name="v_x", grid=model.grid, time_order=2, space_order=s_o+2,
                      staggered=x, coefficients='symbolic')
    vy = TimeFunction(name="v_y", grid=model.grid, time_order=2, space_order=s_o+2,
                      staggered=y, coefficients='symbolic')
    vz = TimeFunction(name="v_z", grid=model.grid, time_order=2, space_order=s_o+2,
                      staggered=z, coefficients='symbolic')

    # Dummy functions
    vx_d = TimeFunction(name='v_x_d', grid=model.grid,
                        space_order=s_o, staggered=x, coefficients='symbolic')
    vy_d = TimeFunction(name='v_y_d', grid=model.grid,
                        space_order=s_o, staggered=y, coefficients='symbolic')
    vz_d = TimeFunction(name='v_z_d', grid=model.grid,
                        space_order=s_o, staggered=z, coefficients='symbolic')

    infile = 'tests/trial_surfaces/angled_surface_'+str(tilt)+'.ply'

    # Zero even pressure derivatives on the boundary
    spec_p = {2*i: 0 for i in range(p.space_order)}
    bcs_p = BoundaryConditions(spec_p, p.space_order)
    # Zero odd derivatives on the boundary
    spec_v = {2*i+1: 0 for i in range(vx_d.space_order)}
    bcs_v = BoundaryConditions(spec_v, vx_d.space_order)

    functions = pd.DataFrame({'function': [p, vx_d, vy_d, vz_d],
                              'bcs': [bcs_p, bcs_v, bcs_v, bcs_v],
                              'subs_function': [None, vx, vy, vz]},
                             columns=['function', 'bcs', 'subs_function'])

    # Create the immersed boundary surface
    surface = ImmersedBoundary('topography', infile, functions,
                               interior_point=tuple(src.coordinates.data[0]),
                               qc=qc, toggle_normals=toggle_normals)

    # Configure derivative needed
    derivs = pd.DataFrame({'function': [p, vx_d, vy_d, vz_d],
                           'derivative': [1, 1, 1, 1],
                           'eval_offset': [(0.5, 0.5, 0.5), (-0.5, 0.5, 0.5),
                                           (0.5, -0.5, 0.5), (0.5, 0.5, -0.5)]},
                          columns=['function', 'derivative', 'eval_offset'])

    more_derivs = pd.DataFrame({'function': [p],
                                'derivative': [2],
                                'eval_offset': [(0., 0., 0.)]},
                               columns=['function', 'derivative',
                                        'eval_offset'])

    coeffs = surface.subs(derivs)
    more_coeffs = surface.subs(more_derivs)

    # Set up AxialDistanceFunction and conditional dimensions so that the
    # 2nd-order pressure update is applied when |eta| < 0.5
    ax = AxialDistanceFunction(p, infile,
                               toggle_normals=toggle_normals)

    # Needed to sidestep a compilation-preventing bug in Devito
    ax_x = Function(name='ax_x', shape=model.grid.shape, dimensions=(x, y, z))
    ax_y = Function(name='ax_y', shape=model.grid.shape, dimensions=(x, y, z))
    ax_z = Function(name='ax_z', shape=model.grid.shape, dimensions=(x, y, z))

    ax_x.data[:] = np.array(ax.axial[0].data[:])
    ax_y.data[:] = np.array(ax.axial[1].data[:])
    ax_z.data[:] = np.array(ax.axial[2].data[:])

    second_update = sp.Or(sp.Or(Le(sp.Abs(ax_x), 0.5*h_x),
                                Le(sp.Abs(ax_y), 0.5*h_y)),
                          Le(sp.Abs(ax_z), 0.5*h_z))

    # Conditional masks for update
    use_2nd = ConditionalDimension(name='use_2nd', parent=z,
                                   condition=second_update)
    use_1st = ConditionalDimension(name='use_1st', parent=z,
                                   condition=sp.Not(second_update))

    # We can now write the system of PDEs
    eq_vx = Eq(vx.forward, vx - dt*p.dx/rho, coefficients=coeffs)
    eq_vy = Eq(vy.forward, vy - dt*p.dy/rho, coefficients=coeffs)
    eq_vz = Eq(vz.forward, vz - dt*p.dz/rho, coefficients=coeffs)
    eq_p = Eq(p.forward,
              p - dt*vel**2*rho*(vx.forward.dx + vy.forward.dy + vz.forward.dz)/(1+model.damp), coefficients=coeffs, implicit_dims=use_1st)
    eq_p2 = Eq(p.forward,
               2*p - p.backward + dt**2*vel**2*p.laplace/(1+model.damp),
               coefficients=more_coeffs, implicit_dims=use_2nd)

    # Finally we define the source injection and receiver read function
    src_term = src.inject(field=p.forward,
                          expr=src)

    # Create interpolation expression for receivers
    rec_term = rec.interpolate(expr=p.forward)

    op = Operator([eq_vx, eq_vy, eq_vz, eq_p, eq_p2] + src_term + rec_term)
    op(time=time_range.num-1, dt=model.critical_dt/2)

    """
    plt.imshow(p.data[-1, :, 60].T, origin='upper')
    plt.colorbar()
    plt.show()
    """

    return rec.data


class TestGathers:
    """
    A class for testing the accuracy of gathers resulting from a reflection off
    the immersed boundary.
    """

    @pytest.mark.parametrize('s_o', [4, 6])
    @pytest.mark.parametrize('spec', [(5, False), (10, True), (15, True),
                                      (20, True), (25, True), (30, True),
                                      (35, True), (40, True), (45, False)])
    def test_tilted_boundary(self, s_o, spec):
        """
        Check that gathers for a tilted boundary match those generated with a
        conventional horizontal free surface and the same geometry.
        """
        tilt, toggle_normals = spec
        max_thres = 0.0026
        avg_thres = 2.5e-4

        # Define a physical size
        shape = (101, 101, 101)  # Number of grid point (nx, ny, nz)
        spacing = (10., 10., 10.)  # Grid spacing in m. The domain size is 1x1x1km
        origin = (0., 0., 0.)

        v = 1.5

        model = Model(vp=v, origin=origin, shape=shape, spacing=spacing,
                      space_order=4, nbl=10, bcs="damp")

        t0 = 0.  # Simulation starts a t=0
        tn = 500.  # Simulation last 0.5 seconds (500 ms)
        dt = model.critical_dt/2  # Time step from model grid spacing

        time_range = TimeAxis(start=t0, stop=tn, step=dt)

        f0 = 0.010  # Source peak frequency is 10Hz (0.010 kHz)

        ref = reference_shot(model, time_range, f0, s_o)
        tilted = tilted_shot(model, time_range, f0, s_o, tilt,
                             toggle_normals=toggle_normals, qc=False)

        """
        plt.imshow(ref, aspect='auto', vmin=-0.03, vmax=0.03, cmap='seismic')
        plt.colorbar()
        plt.show()

        plt.imshow(tilted, aspect='auto', vmin=-0.03, vmax=0.03, cmap='seismic')
        plt.colorbar()
        plt.show()

        plt.imshow(tilted-ref, aspect='auto', vmin=-0.03, vmax=0.03, cmap='seismic')
        plt.colorbar()
        plt.show()
        """

        assert np.amax(np.absolute(ref - tilted)) < max_thres
        assert np.mean(np.absolute(ref-tilted)) < avg_thres


def main(s_o, tilt, filepath, toggle_normals=True, qc=True):
    # Define a physical size
    shape = (101, 101, 101)  # Number of grid point (nx, ny, nz)
    spacing = (10., 10., 10.)  # Grid spacing in m. The domain size is 1x1x1km
    origin = (0., 0., 0.)

    v = 1.5

    model = Model(vp=v, origin=origin, shape=shape, spacing=spacing,
                  space_order=4, nbl=10, bcs="damp")

    t0 = 0.  # Simulation starts a t=0
    tn = 500.  # Simulation last 0.5 seconds (500 ms)
    dt = model.critical_dt/2  # Time step from model grid spacing

    time_range = TimeAxis(start=t0, stop=tn, step=dt)

    f0 = 0.010  # Source peak frequency is 10Hz (0.010 kHz)

    ref = reference_shot(model, time_range, f0, s_o)
    tilted = tilted_shot(model, time_range, f0, s_o, tilt,
                         toggle_normals=toggle_normals, qc=qc)

    fig, (ax0, ax1, ax2) = plt.subplots(nrows=1, ncols=3, figsize=(20, 6))

    im0 = ax0.imshow(ref, aspect='auto', cmap='seismic', extent=[-500, 500, tn, t0],
                     vmin=-0.03, vmax=0.03)  # noqa: F841
    ax0.set_title("Reference")
    ax0.set_xlabel("Offset (m)")
    ax0.set_ylabel("Time (ms)")

    im1 = ax1.imshow(tilted, aspect='auto', cmap='seismic', extent=[-500, 500, tn, t0],
                     vmin=-0.03, vmax=0.03)  # noqa: F841
    ax1.set_title("{} degree tilt".format(str(tilt)))
    ax1.set_xlabel("Offset (m)")

    im2 = ax2.imshow(ref - tilted, aspect='auto', cmap='seismic', extent=[-500, 500, tn, t0],
                     vmin=-0.03, vmax=0.03)  # noqa: F841
    ax2.set_title("Difference")
    ax2.set_xlabel("Offset (m)")
    fig.colorbar(im2)
    fig.tight_layout()
    if filepath == 'show':
        plt.show()
    else:
        plt.savefig(filepath + "/order_{}_tilt_{}".format(str(s_o), str(tilt)))
        plt.close()


if __name__ == "__main__":
    s_o = int(sys.argv[1])
    tilt = int(sys.argv[2])
    filepath = sys.argv[3]
    toggle_normals = bool(int(sys.argv[4]))
    qc = bool(int(sys.argv[5]))
    main(s_o, tilt, filepath, toggle_normals=toggle_normals, qc=qc)
