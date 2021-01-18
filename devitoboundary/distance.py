"""
A module containing classes for the calculation of axial distances from signed
distance functions discretized to a Devito grid.
"""

import numpy as np
import sympy as sp

from devito import Function, VectorFunction, grad, ConditionalDimension, \
    Le, Lt, Gt, Eq, Operator, Grid
from devitoboundary import SDFGenerator

__all__ = ['SignedDistanceFunction', 'AxialDistanceFunction']


class SignedDistanceFunction:
    """
    A class to carry out the calculation of signed distances from a boundary
    surface represented by a point cloud or polygon file within an area of
    effect determined by the specified set of Devito functions.

    Parameters
    ----------
    function : Function
        The Devito function used for configuring distance calculation.
    infile : str
        The path to the input point cloud or polygon file
    offset : tuple of float
        The offset for each axis of the function. Default is (0., 0., 0.).
    toggle_normals : bool
        Flip the direction of the estimated point normals. This has the effect
        of reversing the side of the surface which is considered to be the
        interior. Default is False.

    Attributes
    ----------
    sdf : Devito Function
        The signed distance function.
    grid : Devito Grid
        The grid onto which the signed distance function is discretized.
    """
    def __init__(self, function, infile, offset=(0., 0., 0.),
                 toggle_normals=False):
        self._grid = function.grid
        # Put single functions in a tuple for consistency
        self._function = function

        # Search radius of the SDF is equal to the function order.
        self._order = self._function.space_order

        self._offset = offset
        # Calculate the signed distance function
        # Radius of M/2+1 grid increments
        radius = self._order//2+1
        self._sdfgen = SDFGenerator(infile, self._grid, radius=radius,
                                    offset=self._offset,
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
    function : Function  or tuple of Functions
        The Devito functions used for configuring distance calculation. These
        must share a grid.
    infile : str
        The path to the input point cloud or polygon file
    offset : tuple of float
        The offset for each axis of the function. Default is (0., 0., 0.).
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
    axial : Devito VectorFunction
        The axial distances to the boundary surface.
    """
    def __init__(self, function, infile, offset=(0., 0., 0.),
                 toggle_normals=False):
        super().__init__(function, infile, offset=offset,
                         toggle_normals=toggle_normals)

        # Grid with M/2 nodes of padding
        self._pad = self._pad_grid()

        # Axial signed distance function (note the padded grid)
        self._axial = VectorFunction(name='axial', grid=self._pad,
                                     space_order=self._order,
                                     staggered=(None, None, None))

        self._axial_setup()

    def _axial_setup(self):
        """
        Update the axial distance function from the signed distance function.
        """
        # Recurring value for tidiness
        m_size = int(self._order/2)

        # Create a padded version of the signed distance function
        pad_sdf = Function(name='pad_sdf', grid=self._pad, space_order=self._order)
        pad_sdf.data[:] = np.pad(self._sdf.data, (m_size,), 'edge')

        # Set default values for axial distance
        self._axial[0].data[:] = -self._order*self._pad.spacing[0]
        self._axial[1].data[:] = -self._order*self._pad.spacing[1]
        self._axial[2].data[:] = -self._order*self._pad.spacing[2]

        # Equations to decompose distance into axial distances
        x, y, z = self._pad.dimensions
        h_x, h_y, h_z = self._pad.spacing
        pos = sp.Matrix([x*h_x, y*h_y, z*h_z])

        sdf_grad = grad(pad_sdf).evaluate  # Gradient of the sdf

        # Plane eq: a*x + b*y + c*z = d
        a = sdf_grad[0]
        b = sdf_grad[1]
        c = sdf_grad[2]
        d = sdf_grad.dot(pos - pad_sdf*sdf_grad)

        # Only need to calculate adjacent to boundary
        close_sdf = Le(sp.Abs(pad_sdf), h_x)

        # Also only want values smaller than one increment
        small_x = sp.And(Lt((d - b*pos[1] - c*pos[2])/a - pos[0], h_x),
                         Gt((d - b*pos[1] - c*pos[2])/a - pos[0], -h_x))
        small_y = sp.And(Lt((d - a*pos[0] - c*pos[2])/b - pos[1], h_y),
                         Gt((d - a*pos[0] - c*pos[2])/b - pos[1], -h_y))
        small_z = sp.And(Lt((d - a*pos[0] - b*pos[1])/c - pos[2], h_z),
                         Gt((d - a*pos[0] - b*pos[1])/c - pos[2], -h_z))

        # Conditional mask for calculation
        mask_x = ConditionalDimension(name='mask_x', parent=z,
                                      condition=sp.And(close_sdf, small_x))
        mask_y = ConditionalDimension(name='mask_y', parent=z,
                                      condition=sp.And(close_sdf, small_y))
        mask_z = ConditionalDimension(name='mask_z', parent=z,
                                      condition=sp.And(close_sdf, small_z))

        eq_x = Eq(self._axial[0], (d - b*pos[1] - c*pos[2])/a - pos[0], implicit_dims=mask_x)
        eq_y = Eq(self._axial[1], (d - a*pos[0] - c*pos[2])/b - pos[1], implicit_dims=mask_y)
        eq_z = Eq(self._axial[2], (d - a*pos[0] - b*pos[1])/c - pos[2], implicit_dims=mask_z)

        op_axial = Operator([eq_x, eq_y, eq_z], name='Axial')
        op_axial.apply()

    def _pad_grid(self):
        """
        Return a grid with an additional M/2 nodes of padding on each side vs
        the main grid.
        """
        # Recurring value for tidiness
        m_size = int(self._order/2)
        # Calculate origin position
        p_origin = np.array([value for value in self._grid.origin_map.values()])
        p_origin -= m_size*np.array(self._grid.spacing)
        p_origin = tuple(p_origin)
        # Calculate size and extent of the padded grid
        p_extent = np.array(self._grid.extent)
        p_extent += 2*m_size*np.array(self._grid.spacing)
        p_extent = tuple(p_extent)
        p_shape = np.array(self._grid.shape)+2*m_size
        p_shape = tuple(p_shape)

        p_grid = Grid(shape=p_shape, extent=p_extent, origin=p_origin)

        return p_grid

    @property
    def axial(self):
        """Get the axial distances"""
        m_size = int(self._order/2)
        axial = VectorFunction(name='axial', grid=self._grid,
                               space_order=self._order,
                               staggered=(None, None, None))
        axial[0].data[:] = self._axial[0].data[m_size:-m_size,
                                               m_size:-m_size,
                                               m_size:-m_size]
        axial[1].data[:] = self._axial[1].data[m_size:-m_size,
                                               m_size:-m_size,
                                               m_size:-m_size]
        axial[2].data[:] = self._axial[2].data[m_size:-m_size,
                                               m_size:-m_size,
                                               m_size:-m_size]
        return axial
