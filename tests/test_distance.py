import pytest

import numpy as np
import sympy as sp

from devito import Grid, Function, Gt, ConditionalDimension, Eq, Operator, \
    VectorFunction, Ge, Le
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
        thres = 0.15  # 0.15 grid increments
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

    @pytest.mark.parametrize('space_order', space_order)
    def test_axial_distance(self, space_order):
        """Check the mean error in axial distances"""
        # Error threshold
        thres = 0.3  # 0.3 grid increments
        # Would like this to be lower, but errors in initial distance field
        # due to sampling of the sphere get amplified by higher order schemes
        # Sphere of radius 40, center (50, 50, 50)
        sphere = 'tests/trial_surfaces/sphere.ply'

        # Grid configuration
        extent = (100., 100., 100.)
        shape = (101, 101, 101)
        origin = (0., 0., 0.)

        grid = Grid(shape=shape, extent=extent, origin=origin)
        x, y, z = grid.dimensions
        h_x, h_y, h_z = grid.spacing

        f = VectorFunction(name='f', grid=grid, space_order=space_order,
                           staggered=(None, None, None))

        # Create signed distance function
        ax = AxialDistanceFunction(f, sphere)

        # Positive analytic surface positions
        x_p = 50 + sp.sqrt(40**2 - (y*h_y - 50)**2 - (z*h_z - 50)**2)
        y_p = 50 + sp.sqrt(40**2 - (x*h_x - 50)**2 - (z*h_z - 50)**2)
        z_p = 50 + sp.sqrt(40**2 - (x*h_x - 50)**2 - (y*h_y - 50)**2)

        # Negative analytic surface positions
        x_n = 50 - sp.sqrt(40**2 - (y*h_y - 50)**2 - (z*h_z - 50)**2)
        y_n = 50 - sp.sqrt(40**2 - (x*h_x - 50)**2 - (z*h_z - 50)**2)
        z_n = 50 - sp.sqrt(40**2 - (x*h_x - 50)**2 - (y*h_y - 50)**2)

        # If x > 50, then positive, if x < 50, then negative
        # If the contents of the square root are zero, then undefined
        cond_x_p = sp.And(Ge(x*h_x, 50),
                          Ge(40**2 - (y*h_y - 50)**2 - (z*h_z - 50)**2, 0))
        cond_y_p = sp.And(Ge(y*h_y, 50),
                          Ge(40**2 - (x*h_x - 50)**2 - (z*h_z - 50)**2, 0))
        cond_z_p = sp.And(Ge(z*h_z, 50),
                          Ge(40**2 - (x*h_x - 50)**2 - (y*h_y - 50)**2, 0))

        cond_x_n = sp.And(Le(x*h_x, 50),
                          Ge(40**2 - (y*h_y - 50)**2 - (z*h_z - 50)**2, 0))
        cond_y_n = sp.And(Le(y*h_y, 50),
                          Ge(40**2 - (x*h_x - 50)**2 - (z*h_z - 50)**2, 0))
        cond_z_n = sp.And(Le(z*h_z, 50),
                          Ge(40**2 - (x*h_x - 50)**2 - (y*h_y - 50)**2, 0))

        # Only care about values where axial function is evaluated
        eval_x = Gt(ax.axial[0], -space_order)
        eval_y = Gt(ax.axial[1], -space_order)
        eval_z = Gt(ax.axial[2], -space_order)

        # Set up conditional dimensions
        mask_x_p = ConditionalDimension(name='mask_x_p', parent=z,
                                        condition=sp.And(cond_x_p, eval_x))
        mask_y_p = ConditionalDimension(name='mask_y_p', parent=z,
                                        condition=sp.And(cond_y_p, eval_y))
        mask_z_p = ConditionalDimension(name='mask_z_p', parent=z,
                                        condition=sp.And(cond_z_p, eval_z))

        mask_x_n = ConditionalDimension(name='mask_x_n', parent=z,
                                        condition=sp.And(cond_x_n, eval_x))
        mask_y_n = ConditionalDimension(name='mask_y_n', parent=z,
                                        condition=sp.And(cond_y_n, eval_y))
        mask_z_n = ConditionalDimension(name='mask_z_n', parent=z,
                                        condition=sp.And(cond_z_n, eval_z))

        # Set up equations
        eq_x_p = Eq(f[0], ax.axial[0] + x*h_x - x_p, implicit_dims=mask_x_p)
        eq_y_p = Eq(f[1], ax.axial[1] + y*h_y - y_p, implicit_dims=mask_y_p)
        eq_z_p = Eq(f[2], ax.axial[2] + z*h_z - z_p, implicit_dims=mask_z_p)

        eq_x_n = Eq(f[0], ax.axial[0] + x*h_x - x_n, implicit_dims=mask_x_n)
        eq_y_n = Eq(f[1], ax.axial[1] + y*h_y - y_n, implicit_dims=mask_y_n)
        eq_z_n = Eq(f[2], ax.axial[2] + z*h_z - z_n, implicit_dims=mask_z_n)

        eqns = [eq_x_p, eq_x_n, eq_y_p, eq_y_n, eq_z_p, eq_z_n]
        op = Operator(eqns, name='ErrCalc')
        op.apply()

        # Calculate mean error
        err_x = np.absolute(f[0].data[f[0].data != 0])
        avg_x = np.mean(err_x)

        err_y = np.absolute(f[1].data[f[1].data != 0])
        avg_y = np.mean(err_y)

        err_z = np.absolute(f[2].data[f[2].data != 0])
        avg_z = np.mean(err_z)

        if avg_x > thres:
            err_message = "Mean error in x distances is {:.6}"
            raise ValueError(err_message.format(avg_x))

        if avg_y > thres:
            err_message = "Mean error in y distances is {:.6}"
            raise ValueError(err_message.format(avg_y))

        if avg_z > thres:
            err_message = "Mean error in z distances is {:.6}"
            raise ValueError(err_message.format(avg_z))

# class TestFunctions
