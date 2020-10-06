import pytest

import numpy as np
import sympy as sp

from devito import Grid, Function, Gt, ConditionalDimension, Eq, Operator
from devitoboundary import SignedDistanceFunction


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
        thres = 0.15  # 0.1 grid increments
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
        def_sdf = -int(space_order/2) - 1

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


# class TestFunctions
