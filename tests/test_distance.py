import pytest

import numpy as np
import sympy as sp

from devito import Grid, Function, Gt, ConditionalDimension, Eq, Operator
from devitoboundary import SignedDistanceFunction, AxialDistanceFunction


class TestError:
    """
    A class to test the error in distance measurement against a sampled sphere.
    Errors are calculated against the analytical function for that sphere.
    """

    space_order = [2, 4, 6, 8]

    @pytest.mark.parametrize('space_order', space_order)
    def test_sdf_distance(self, space_order):
        """Check the mean error in SDF values"""
        # Error threshold
        thres = 0.11  # 0.11 grid increments
        # Sphere of radius 40, center (50, 50, 50)
        sphere = 'tests/trial_surfaces/sphere.ply'

        # Grid configuration
        extent = (100., 100., 100.)
        shape = (101, 101, 101)
        origin = (0., 0., 0.)

        grid = Grid(shape=shape, extent=extent, origin=origin)
        x, y, z = grid.dimensions
        h_x, h_y, h_z = grid.spacing

        f = Function(name='f', grid=grid, space_order=space_order)

        # Create signed distance function
        sig = SignedDistanceFunction(f, sphere)

        # Default sdf value
        def_sdf = -space_order//2 - 1

        # Only care about areas where sdf has been evaluated
        cond = Gt(sig.sdf, def_sdf)
        mask = ConditionalDimension(name='mask', parent=z, condition=cond)

        # Distance to boundary is radius minus distance from center
        r = sp.sqrt((h_x*x - 50)**2 + (h_y*y - 50)**2 + (h_z*z - 50)**2)
        # Evaluate absolute error (in grid increments)
        eq = Eq(f, sp.Abs(sig.sdf - 40 + r), implicit_dims=mask)

        op = Operator(eq, name='ErrCalc')
        op.apply()

        # Calculate mean error
        err = f.data[f.data != 0]
        avg = np.mean(err)

        if avg > thres:
            err_message = "Mean error in measured distances is {:.6}"
            raise ValueError(err_message.format(avg))

    @pytest.mark.parametrize('space_order', space_order)
    def test_axial_distance(self, space_order):
        """Check the mean error in axial distances"""
        # Error threshold
        thres = 0.3  # 0.3 grid increments
        # Would like this to be lower, but errors in initial distance field
        # due to sampling of the sphere get amplified by higher order schemes
        # Also curvature of sphere is very large near poles.
        # Sphere of radius 40, center (50, 50, 50)
        sphere = 'tests/trial_surfaces/sphere.ply'

        # Grid configuration
        extent = (100., 100., 100.)
        shape = (101, 101, 101)
        origin = (0., 0., 0.)

        grid = Grid(shape=shape, extent=extent, origin=origin)
        x, y, z = grid.dimensions
        h_x, h_y, h_z = grid.spacing

        f = Function(name='f', grid=grid, space_order=space_order)

        # Create signed distance function
        ax = AxialDistanceFunction(f, sphere)

        locs_xx, locs_xy, locs_xz = np.where(ax.axial[0].data != -space_order)
        locs_yx, locs_yy, locs_yz = np.where(ax.axial[1].data != -space_order)
        locs_zx, locs_zy, locs_zz = np.where(ax.axial[2].data != -space_order)

        # Analytic surface positions
        def true_dist(pos_1, pos_2, pos_3, positive):
            if positive:
                return -pos_1*h_x + 50 + np.sqrt(40**2 - (pos_2*h_y-50)**2 - (pos_3*h_z-50)**2)
            else:
                return -pos_1*h_x + 50 - np.sqrt(40**2 - (pos_2*h_y-50)**2 - (pos_3*h_z-50)**2)

        mask_x = locs_xx > 50
        data_x = ax.axial[0].data[locs_xx, locs_xy, locs_xz]
        true_xp = true_dist(locs_xx, locs_xy, locs_xz, True)
        true_xn = true_dist(locs_xx, locs_xy, locs_xz, False)
        err_x = np.absolute(data_x - np.where(mask_x, true_xp, true_xn))

        mask_y = locs_yy > 50
        data_y = ax.axial[1].data[locs_yx, locs_yy, locs_yz]
        true_yp = true_dist(locs_yy, locs_yx, locs_yz, True)
        true_yn = true_dist(locs_yy, locs_yx, locs_yz, False)
        err_y = np.absolute(data_y - np.where(mask_y, true_yp, true_yn))

        mask_z = locs_zz > 50
        data_z = ax.axial[2].data[locs_zx, locs_zy, locs_zz]
        true_zp = true_dist(locs_zz, locs_zx, locs_zy, True)
        true_zn = true_dist(locs_zz, locs_zx, locs_zy, False)
        err_z = np.absolute(data_z - np.where(mask_z, true_zp, true_zn))

        mean_err = np.nanmean(np.concatenate([err_x, err_y, err_z]))

        if mean_err > thres:
            message = "Mean error greater than threshold: {}"
            raise ValueError(message.format(mean_err))
