"""
A module containing classes for the calculation of axial distances from signed
distance functions discretized to a Devito grid.
"""

import sympy as sp

from devito import Function, VectorFunction, grad, ConditionalDimension, \
    Le, Gt, Eq, Operator
from devitoboundary import SDFGenerator

__all__ = ['SignedDistanceFunction', 'AxialDistanceFunction']


class SignedDistanceFunction:
    """
    A class to carry out the calculation of signed distances from a boundary
    surface represented by a point cloud or polygon file within an area of
    effect determined by the specified set of Devito functions.

    Parameters
    ----------
    functions : Function  or tuple of Functions
        The Devito functions used for configuring distance calculation. These
        must share a grid.
    infile : str
        The path to the input point cloud or polygon file
    toggle_normals : bool
        Flip the direction of the estimated point normals. This has the effect
        of reversing the side of the surface which is considered to be the
        interior. Default is False.

    Attributes
    ----------
    function : Devito Function
        The signed distance function.
    grid : Devito Grid
        The grid onto which the signed distance function is discretized.
    """
    def __init__(self, functions, infile, toggle_normals=False):
        # Check variable type
        is_tuple = isinstance(functions, tuple)
        is_function = issubclass(type(functions), Function) \
            or issubclass(type(functions), VectorFunction)

        if is_tuple:
            # Multiple functions supplied
            for function in functions:
                if function.grid is not functions[0].grid:
                    grid_err = "Functions do not share a grid."
                    raise ValueError(grid_err)
            self._grid = functions[0].grid
            self._functions = functions
        elif is_function:
            self._grid = functions.grid
            # Put single functions in a tuple for consistency
            self._functions = (functions,)

        # Search radius of the SDF is equal to the highest function order.
        self._order = max([function.space_order for function in self._functions])
        # Calculate the signed distance function
        self._sdfgen = SDFGenerator(infile, self._grid, radius=self._order,
                                    toggle_normals=toggle_normals)

        # Create a Devito function to store and manipulate the sdf
        self._sdf = Function(name='sdf', grid=self._grid, space_order=self._order)
        self._sdf.data[:] = self._sdfgen.array

    @property
    def sdf(self):
        """Get the absolute signed distance function."""
        return self._sdf

    @property
    def grid(self):
        """Get the grid on which the function is discretized"""
        return self._grid


class AxialDistanceFunction(SignedDistanceFunction):
    """
    A class to carry out the calculation of signed distances along each axis
    from a boundary surface represented by a point cloud or polygon file within
    an area of effect determined by the specified set of Devito functions.

    Parameters
    ----------
    functions : Function  or tuple of Functions
        The Devito functions used for configuring distance calculation. These
        must share a grid.
    infile : str
        The path to the input point cloud or polygon file
    toggle_normals : bool
        Flip the direction of the estimated point normals. This has the effect
        of reversing the side of the surface which is considered to be the
        interior. Default is False.

    Attributes
    ----------
    function : Devito Function
        The signed distance function.
    grid : Devito Grid
        The grid onto which the signed distance function is discretized.
    """
    def __init__(self, functions, infile, toggle_normals=False):
        super().__init__(functions, infile, toggle_normals)

        # Will need some kind of conditional dimension.

        # The axial signed distance function
        self._axial = VectorFunction(name='axial', grid=self._grid,
                                     space_order=self._order,
                                     staggered=(None, None, None))
        self._axial_setup()

    def _axial_setup(self):
        """
        Update the axial distance function from the signed distance function.
        """
        # Set default values for axial distance
        # self._axial[0].data[:] = -self._order*self._grid.spacing[0]
        # self._axial[1].data[:] = -self._order*self._grid.spacing[1]
        # self._axial[2].data[:] = -self._order*self._grid.spacing[2]

        # Equations to decompose distance into axial distances
        x, y, z = self._grid.dimensions
        h_x, h_y, h_z = self._grid.spacing
        pos = sp.Matrix([x*h_x, y*h_y, z*h_z])

        sdf_grad = grad(self._sdf).evaluate  # Gradient of the sdf

        # a*x + b*y + c*z = d
        a = sdf_grad[0]
        b = sdf_grad[1]
        c = sdf_grad[2]
        # d = sdf_grad.dot(pos + self._sdf*sdf_grad)
        d = sdf_grad.dot(pos - self._sdf*sdf_grad)

        # Only need to calculate where SDF <= dx*M/2 (otherwise out of AOE)
        # close_sdf = Le(self._sdf, h_x*self._order/2)
        close_sdf = Le(sp.Abs(self._sdf), h_x*self._order/2)
        # If any of these are very small, then distance is very large
        close_x = Gt(sp.Abs(a), 1e-6)
        close_y = Gt(sp.Abs(b), 1e-6)
        close_z = Gt(sp.Abs(c), 1e-6)

        # Conditional mask for calculation
        # FIXME: Consolidate these
        mask_x = ConditionalDimension(name='mask_x', parent=z,
                                      condition=close_sdf)
        mask_y = ConditionalDimension(name='mask_y', parent=z,
                                      condition=close_sdf)
        mask_z = ConditionalDimension(name='mask_z', parent=z,
                                      condition=close_sdf)

        mask_test = ConditionalDimension(name='mask_test', parent=z,
                                         condition=close_sdf)

        # Need some mins or something here.
        eq_x = Eq(self._axial[0], (d - b*pos[1] - c*pos[2])/a - pos[0], implicit_dims=mask_x)
        eq_y = Eq(self._axial[1], (d - a*pos[0] - c*pos[2])/b - pos[1], implicit_dims=mask_y)
        eq_z = Eq(self._axial[2], (d - a*pos[0] - b*pos[1])/c - pos[2], implicit_dims=mask_z)

        # eq_test = Eq(self._axial, sdf_grad, implicit_dims=mask_test)
        # eq_test = Eq(self._axial, -self._sdf*sdf_grad, implicit_dims=mask_test)
        # op_axial = Operator([eq_test])

        # op_axial = Operator([eq_x])

        op_axial = Operator([eq_x, eq_y, eq_z], name='axial')
        # print(op_axial.ccode)
        op_axial.apply()

    @property
    def axial(self):
        """Get the axial distances"""
        return self._axial
